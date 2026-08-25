from __future__ import annotations

import contextlib
import copy
import math
from typing import Optional, Sequence, Tuple
from icecream import ic

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .utils.attention import CrossAttentionTransformer, GatedRecurrentCore
from .utils.rope import RoPE, axial_rope, rope_periods


class RecurrentWorldModel(nn.Module):
    """Shape symbols used throughout:

        B   batch                 N   frames per clip
        P   patches per frame     D   encoder width      Dd  decoder width
        Tt  target frames         L   = N * (1 + P)  flattened memory length

    Token axis is always [CLS, patch_0 .. patch_{P-1}], so width is 1 + P.

    Positional encoding
    -------------------
    Every token -- context and query alike -- carries a coordinate triple
    (tau, i, j): absolute timestamp, and normalized spatial position on a
    [-1, +1] grid so that changing input resolution does not change the
    relative distance between patches. These are injected as a 3-axial rotary
    embedding inside each attention block, following DINO-world (Baldassarre
    et al., 2025, Sec. 3.2). The query token is a single learnable vector with
    no positional content of its own; two queries for different patches of the
    same target frame start out as literally the same vector and are separated
    only by their rotations.

    Because (R(a) q) . (R(b) k) = q . R(b - a) k, absolute coordinates on both
    sides yield relative offsets in the logits. That is what tags the memory
    with per-frame time for free, which an additive query-side gap cannot do.

    CLS sits at spatial (0, 0) -- the grid centre -- with its frame's real tau.
    It is distinguished from the centre patch by content (`cls_query`), not by
    rotation. Passing `drop_cls=True` reproduces DINO-world exactly, which
    discards CLS and registers and keeps only patch tokens.

    Args:
        encoder:        pre-built HF backbone; must expose .embeddings,
                        .encoder, .layernorm, .config. Downloads if None.
        freeze_encoder: freeze the encoder and use it directly for targets
                        (detached). Cheapest and safest starting point.
        use_ema:        if the encoder is trainable, keep an EMA copy to
                        produce targets. Without one of freeze_encoder or
                        use_ema, gradients reach both sides of the loss and
                        the representation can collapse.
        context_mode:   'full'  -> attend every token of every earlier frame
                                   (L = N*(1+P)), causality via mask.
                        'state' -> attend only the GRU state after frame t-1
                                   (L = 1+P). Strictly causal by construction,
                                   far cheaper, no mask needed. That state is
                                   tagged tau_{t-1}, i.e. "the summary as of
                                   the last observed frame".
        rope_time_periods: (fastest, slowest) period in the SAME UNITS as
                        `frame_times`. The default suits gaps of ~1-12 with
                        clips spanning ~100. If you feed seconds, DINO-world's
                        range is (1e-2, 1e2).
        rope_space_periods: periods in normalized grid units. Range is 2.0
                        edge-to-edge, one patch is 2/grid, so ~(0.2, 4.0)
                        covers neighbour-level to whole-image structure.
        normalize_target: layer-norm each target token (over D) before the
                        MSE. Without it, a handful of high-norm tokens --
                        CLS, and register-less DINOv2's high-norm artifact
                        patches -- dominate the mean over all tokens, since
                        F.mse_loss averages raw squared error across every
                        token indiscriminately. Per-token layer-norm puts
                        every token on the same scale before the loss sees
                        it; `pred` needs no matching treatment since
                        `repr_head` is free to learn output at that scale.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_name: str = "facebook/dinov2-with-registers-small",
        core_layers: int = 4,
        core_heads: int = 8,
        core_mlp: Optional[int] = None,
        dec_dim: int = 512,
        dec_layers: int = 3,
        dec_heads: int = 16,
        dec_mlp: int = 2048,
        freeze_encoder: bool = True,
        use_ema: bool = False,
        ema_momentum: float = 0.998,
        context_mode: str = "full",
        checkpoint_core: bool = False,
        loss_beta: float = 0.1,
        rope_time_periods: Tuple[float, float] = (1.0, 200.0),
        rope_space_periods: Tuple[float, float] = (0.2, 4.0),
        drop_cls: bool = False,
        normalize_target: bool = False,
        pixel_recon: bool = False,
        objective: str = "repr",
    ):
        super().__init__()
        assert context_mode in ("full", "state")
        assert objective in ("repr", "pixel")
        assert objective != "pixel" or pixel_recon, "objective='pixel' needs pixel_recon=True"

        if encoder is None:
            from transformers import AutoModel
            print(f'loading the vision encoder from {encoder_name}')
            encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder = encoder

        self.d_enc = encoder.config.hidden_size
        self.patch = encoder.config.patch_size
        # 0 for encoders without registers (e.g. plain dinov2-small).
        self.num_register_tokens = getattr(encoder.config, "num_register_tokens", 0)
        self.dec_dim = dec_dim
        self.freeze_encoder = freeze_encoder
        self.context_mode = context_mode
        self.checkpoint_core = checkpoint_core
        self.ema_momentum = ema_momentum
        self.loss_beta = loss_beta
        self.t_periods = rope_time_periods
        self.s_periods = rope_space_periods
        self.drop_cls = drop_cls
        self.normalize_target = normalize_target
        self.pixel_recon = pixel_recon
        self.objective = objective

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

        self.use_ema = use_ema and not freeze_encoder
        if self.use_ema:
            self.target_encoder = copy.deepcopy(encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
            self.target_encoder.eval()
        else:
            self.target_encoder = None

        core_mlp = core_mlp or 4 * self.d_enc
        self.core = GatedRecurrentCore(self.d_enc, core_heads, core_layers, core_mlp)

        self.decoder_embed = nn.Linear(self.d_enc, dec_dim)
        self.decoder = CrossAttentionTransformer(dec_dim, dec_heads, dec_layers, dec_mlp)
        self.repr_head = nn.Linear(dec_dim, self.d_enc)

        # reconstructs pixels from the decoder's output (see `forward`). Whether this
        # trains `core`/`decoder` (objective='pixel') or is just a detached probe
        # that watches them without shaping them (objective='repr') is decided in
        # `forward`. Small on purpose: a 2-layer MLP per patch token, not a real
        # image decoder.
        if self.pixel_recon:
            self.recon_head = nn.Sequential(
                nn.Linear(dec_dim, dec_dim // 2),
                nn.GELU(),
                nn.Linear(dec_dim // 2, self.patch * self.patch * 3),
            )

        # the entire content of a query: one learnable vector, shared by every
        # patch of every target frame. Position enters only through rotation.
        self.query_token = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)
        self.cls_query = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)

    # encoder

    def _maybe_drop_cls(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[..., 1:, :] if self.drop_cls else tokens

    def _drop_registers(self, tokens: torch.Tensor) -> torch.Tensor:
        """HF layout is [CLS, reg_0 .. reg_{R-1}, patch_0 ..]; registers carry
        no spatial position and exist only to soak up the high-norm artifact
        behaviour a register-less DINOv2 dumps into ordinary patch tokens
        (Darcet et al., 2023). Drop them right here so every downstream
        consumer -- core, decoder, loss -- sees the same [CLS, patch_0 ..]
        layout regardless of which encoder variant is loaded."""
        r = self.num_register_tokens
        if r == 0:
            return tokens
        return torch.cat([tokens[..., :1, :], tokens[..., 1 + r:, :]], dim=-2)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(M, 3, H, W) -> (M, 1+P, D) through the online encoder."""
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            h = self.encoder.encoder(self.encoder.embeddings(frames)).last_hidden_state
            h = self._drop_registers(h)
            return self._maybe_drop_cls(self.encoder.layernorm(h))
        

    def _grid(self, P: int, device) -> torch.Tensor:
        """(n_tok, 2) spatial coords on [-1, +1]; CLS (if kept) at the centre.

        Rebuilt each call: it is a few hundred floats, and unlike a cached
        buffer it does not mutate module state, so it is compile- and
        DDP-friendly.
        """
        # ic(P)
        g = int(round(math.sqrt(P)))
        assert g * g == P, f"{P} patch tokens is not a square grid"
        lin = torch.linspace(-1.0, 1.0, g, device=device, dtype=torch.float32)
        ii, jj = torch.meshgrid(lin, lin, indexing="ij")
        grid = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=-1)  # (P, 2)
        if self.drop_cls:
            return grid
        return torch.cat([grid.new_zeros(1, 2), grid], dim=0)         # (1+P, 2)

    def _coords(self, tau: torch.Tensor, P: int) -> torch.Tensor:
        """tau (..., ) absolute times -> (..., n_tok, 3) as (tau, i, j)."""
        device = tau.device
        grid = self._grid(P, device)                                  # (n_tok, 2)
        n_tok = grid.shape[0]
        sp = grid.expand(*tau.shape, n_tok, 2)
        t = tau.to(torch.float32).unsqueeze(-1).unsqueeze(-1).expand(*tau.shape, n_tok, 1)
        return torch.cat([t, sp], dim=-1)

    def _unpatchify(self, patches: torch.Tensor, grid: int) -> torch.Tensor:
        """(M, grid*grid, patch*patch*3) -> (M, 3, grid*patch, grid*patch)."""
        p = self.patch
        M = patches.shape[0]
        x = patches.reshape(M, grid, grid, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)          # M, 3, grid, p, grid, p
        return x.reshape(M, 3, grid * p, grid * p)

    def _rope(self, coords: torch.Tensor, head_dim: int) -> RoPE:
        per = rope_periods(head_dim, self.t_periods, self.s_periods, coords.device)
        return axial_rope(coords, per)


    def _run_core(self, feats: torch.Tensor, frame_times: torch.Tensor,
                  state: torch.Tensor, t: int, n_tok: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step with the right time tags on x and on the state."""
        hd = self.core.head_dim
        tau_now = frame_times[:, t]
        tau_prev = frame_times[:, max(t - 1, 0)]
        x_pos = self._rope(self._coords(tau_now, n_tok - (0 if self.drop_cls else 1)), hd)
        s_pos = self._rope(self._coords(tau_prev, n_tok - (0 if self.drop_cls else 1)), hd)
        if self.checkpoint_core and self.training:
            return checkpoint(self.core, feats[:, t], state, x_pos, s_pos,
                              use_reentrant=False)
        return self.core(feats[:, t], state, x_pos, s_pos)

    def forward(
        self,
        frames: torch.Tensor,               # (B, N, 3, H, W)
        target_idx: torch.Tensor,           # (Tt,) long, indices into N, all >= 1
        frame_times: Optional[torch.Tensor] = None,   # (B, N) float
        state: Optional[torch.Tensor] = None,
    ) -> dict:
        B, N = frames.shape[:2]
        img = frames.shape[2:]
        device = frames.device
        target_idx = target_idx.to(device).long() # assume C...N
        Tt = int(target_idx.numel())
        assert int(target_idx.min()) >= 1, "frame 0 has no history; it cannot be a target"
        assert int(target_idx.max()) < N, "target_idx out of range"

        if frame_times is None:                       # contiguous fallback
            frame_times = torch.arange(N, device=device, dtype=torch.float32)
            frame_times = frame_times.unsqueeze(0).expand(B, N)
        frame_times = frame_times.to(device=device, dtype=torch.float32)

        feats = self.encode(frames.reshape(B * N, *img))           # (B*N, n_tok, D)
        feats = feats.view(B, N, *feats.shape[1:])                 # (B, N, n_tok, D)
        # ic(feats.shape)
        n_tok = feats.shape[2]
        P = n_tok if self.drop_cls else n_tok - 1
        dtype = feats.dtype

        if state is None:
            state = feats.new_zeros(B, n_tok, self.d_enc)
        memory = []
        for t in range(N):
            out, state = self._run_core(feats, frame_times, state, t, n_tok)
            memory.append(out)
        memory = torch.stack(memory, dim=1)                        # (B, N, n_tok, D)

        hd = self.decoder.head_dim
        if self.context_mode == "full":
            L = N * n_tok
            kv = self.decoder_embed(memory.reshape(B, L, self.d_enc))  # bs, t, num_p, emb_d --> bs, t * num_p, emb_d
            kv = kv.unsqueeze(1).expand(B, Tt, L, self.dec_dim).reshape(B * Tt, L, self.dec_dim) 

            # memory[:, t] summarizes everything up to and including frame t,
            # so it is tagged with tau_t.
            kv_coords = self._coords(frame_times, P).reshape(B, L, 3)          # (B, L, 3)
            kv_coords = kv_coords.unsqueeze(1).expand(B, Tt, L, 3).reshape(B * Tt, L, 3)

            frame_of_tok = torch.arange(N, device=device).repeat_interleave(n_tok)
            allowed = frame_of_tok.view(1, L) < target_idx.view(Tt, 1)         # (Tt, L)
            attn_mask = (
                allowed.view(1, Tt, 1, 1, L)
                .expand(B, Tt, 1, 1, L)
                .reshape(B * Tt, 1, 1, L)
            )
        else:  # 'state': the GRU state after frame t-1 already is the history
            kv = memory[:, target_idx - 1]                                     # (B,Tt,n_tok,D)
            kv = self.decoder_embed(kv).reshape(B * Tt, n_tok, self.dec_dim)
            kv_coords = self._coords(frame_times[:, target_idx - 1], P)        # (B,Tt,n_tok,3)
            kv_coords = kv_coords.reshape(B * Tt, n_tok, 3)
            attn_mask = None

        q_coords = self._coords(frame_times[:, target_idx], P).reshape(B * Tt, n_tok, 3)
        queries = self._build_queries(B * Tt, P, dtype)

        decoded = self.decoder(
            queries, kv, attn_mask,
            q_pos=self._rope(q_coords, hd), kv_pos=self._rope(kv_coords, hd),
        )
        pred = self.repr_head(decoded).view(B, Tt, n_tok, self.d_enc)

        gap = frame_times[:, target_idx] - frame_times[:, target_idx - 1]

        # `pred`/`repr_loss` are always computed -- cheap (one linear layer) and
        # useful to log even when they are not the training signal.
        repr_loss = self.loss(pred=pred, target=feats, target_idx=target_idx)
        recon_img, recon_loss = None, None

        if self.pixel_recon:
            # objective='pixel': gradients flow into `core`/`decoder` from pixel
            # space, and `loss` below ignores repr_loss entirely -- this replaces
            # the representation objective rather than adding to it.
            # objective='repr': detach, so this is a passive probe on what the
            # decoder already produces, and cannot influence it.
            tokens = decoded if self.objective == "pixel" else decoded.detach()
            # (unlike `encode`'s output, `decoded` already omits CLS when drop_cls is
            # set -- `_build_queries` never appended `cls_query` in that case.)
            patch_tokens = tokens if self.drop_cls else tokens[:, 1:]  # (B*Tt, P, Dd)
            recon_patches = self.recon_head(patch_tokens)              # (B*Tt, P, patch^2*3)
            recon_img = self._unpatchify(recon_patches, grid=int(round(math.sqrt(P))))
            recon_img = recon_img.view(B, Tt, *recon_img.shape[1:])    # (B, Tt, 3, H, W)

            target_img = frames[:, target_idx]                         # (B, Tt, 3, H, W)
            recon_loss = F.mse_loss(recon_img, target_img)

        loss = recon_loss if self.objective == "pixel" else repr_loss

        return {
            "pred": pred,                # (B, Tt, n_tok, D)
            "memory": memory,            # (B, N,  n_tok, D)
            "state": state,              # (B, n_tok, D)
            "gap": gap,                  # (B, Tt), reported only; RoPE carries it
            "grid": int(round(math.sqrt(P))),
            "recon": recon_img,          # (B, Tt, 3, H, W) if pixel_recon else None
            "repr_loss": repr_loss,
            "recon_loss": recon_loss,
            "loss": loss,
        }

    def _build_queries(self, n_rows: int, P: int, dtype) -> torch.Tensor:
        """(n_rows, n_tok, Dd). Pure content -- no additive position at all."""
        q = self.query_token.expand(n_rows, P, self.dec_dim).to(dtype)
        if self.drop_cls:
            return q
        cls = self.cls_query.expand(n_rows, 1, self.dec_dim).to(dtype)
        return torch.cat([cls, q], dim=1)

    def loss(self, pred: torch.Tensor, target: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """target is `feats` (B, N, n_tok, D); pick out the frames pred was built for."""
        tgt = target[:, target_idx, ...].detach()
        if self.normalize_target:
            tgt = F.layer_norm(tgt, tgt.shape[-1:])
        return F.mse_loss(pred, tgt)


    @torch.no_grad()
    def rollout(
        self,
        context: torch.Tensor,           # (B, C, 3, H, W)
        gaps: Sequence[float] | torch.Tensor,   # per-step time gaps to predict
        context_times: Optional[torch.Tensor] = None,   # (B, C) float
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout in latent space. Returns (B, S, n_tok, D).

        Each predicted frame is fed back through the recurrent core in place of
        encoder features, so this is the inference-time path the teacher-forced
        training objective is an approximation of. Expect the gap between the
        two to widen with horizon.

        The positional convention here is identical to `forward`: predicted
        frame s is tagged with the absolute time it would have had. Any
        divergence between the two paths shows up as silent quality loss, so
        both build coordinates through `_coords`.
        """
        self.eval()
        B, C = context.shape[:2]
        device = context.device
        feats = self.encode(context.reshape(B * C, *context.shape[2:]))
        feats = feats.view(B, C, *feats.shape[1:])
        n_tok = feats.shape[2]
        P = n_tok if self.drop_cls else n_tok - 1
        dtype = feats.dtype

        if context_times is None:
            context_times = torch.arange(C, device=device, dtype=torch.float32)
            context_times = context_times.unsqueeze(0).expand(B, C)
        context_times = context_times.to(device=device, dtype=torch.float32)

        if not torch.is_tensor(gaps):
            gaps = torch.tensor(list(gaps), device=device, dtype=torch.float32)
        gaps = gaps.to(device).float()
        # absolute times of the predicted frames
        step_times = context_times[:, -1:] + gaps.cumsum(0).unsqueeze(0)   # (B, S)

        if state is None:
            state = feats.new_zeros(B, n_tok, self.d_enc)
        memory, times = [], []
        for t in range(C):
            out, state = self._run_core(feats, context_times, state, t, n_tok)
            memory.append(out)
            times.append(context_times[:, t])

        hd = self.decoder.head_dim
        preds = []
        for s in range(gaps.numel()):
            tau = step_times[:, s]
            if self.context_mode == "full":
                kv = torch.stack(memory, dim=1).reshape(B, -1, self.d_enc)
                kv_tau = torch.stack(times, dim=1)                        # (B, k)
            else:
                kv = memory[-1]
                kv_tau = times[-1].unsqueeze(1)                           # (B, 1)
            kv = self.decoder_embed(kv)
            kv_coords = self._coords(kv_tau, P).reshape(B, -1, 3)
            q_coords = self._coords(tau, P)

            q = self._build_queries(B, P, dtype)
            step = self.repr_head(self.decoder(
                q, kv, None,
                q_pos=self._rope(q_coords, hd), kv_pos=self._rope(kv_coords, hd),
            ))
            preds.append(step)

            # feed the prediction back, with its own timestamp
            prev_tau = times[-1]
            hdc = self.core.head_dim
            out, state = self.core(
                step, state,
                self._rope(self._coords(tau, P), hdc),
                self._rope(self._coords(prev_tau, P), hdc),
            )
            memory.append(out)
            times.append(tau)

        # return torch.stack(preds, dim=1)
        return {
            "pred": torch.stack(preds, dim=1),
            "context_memory": torch.stack(memory, dim=1),
            "context_feat": feats,
        }

    @torch.no_grad()
    def step(
        self,
        frame: torch.Tensor,                       # (B, 3, H, W)
        state: Optional[torch.Tensor] = None,
        t_now: Optional[torch.Tensor] = None,      # (B,) absolute time
        t_prev: Optional[torch.Tensor] = None,     # (B,) time of the state
    ):
        """Streaming: one frame in, (features, new_state) out."""
        tokens = self.encode(frame)
        B, n_tok = tokens.shape[0], tokens.shape[1]
        P = n_tok if self.drop_cls else n_tok - 1
        if state is None:
            state = tokens.new_zeros(tokens.shape)
        if t_now is None:
            t_now = tokens.new_zeros(B)
        if t_prev is None:
            t_prev = t_now
        hd = self.core.head_dim
        return self.core(
            tokens, state,
            self._rope(self._coords(t_now.to(tokens.device), P), hd),
            self._rope(self._coords(t_prev.to(tokens.device), P), hd),
        )
