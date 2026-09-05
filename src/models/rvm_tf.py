from __future__ import annotations

import contextlib
import copy
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .utils.attention import CrossAttentionTransformer, GatedRecurrentCore
from .utils.rope import RoPE, axial_rope, rope_periods


class CNNAdapter(nn.Module):
    """Projects per-token DINO features into core space with a CNN instead of
    an MLP: a frame's patch tokens are reassembled into their (g, g) spatial
    grid (g = sqrt(P)) so the projection can mix neighbouring patches, then
    channels are reduced from `d_enc` to `out_dim` by two 3x3 convolutions --
    the conv analogue of the old adapter's expand-then-project 2-layer MLP.

    (..., P, d_enc) -> (..., P, out_dim). No CLS handling: P must already be
    a perfect square (patch tokens only).
    """

    def __init__(self, d_enc: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        hidden = 4 * out_dim
        self.norm = nn.LayerNorm(d_enc)
        self.net = nn.Sequential(
            nn.Conv2d(d_enc, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, P, D = x.shape
        g = int(round(math.sqrt(P)))
        assert g * g == P, f"{P} patch tokens is not a square grid"
        x = self.norm(x)
        x = x.reshape(-1, P, D).transpose(1, 2).reshape(-1, D, g, g)  # (M, D, g, g)
        x = self.net(x)                                              # (M, out_dim, g, g)
        x = x.flatten(2).transpose(1, 2)                              # (M, P, out_dim)
        return x.reshape(*lead, P, self.out_dim)


class RecurrentWorldModel(nn.Module):
    """Shape symbols used throughout:

        B   batch                 N   frames per clip
        P   patches per frame     D   encoder width      Dd  decoder width
        Tt  target frames         L   = N * P  flattened memory length

    CLS and register tokens are always dropped right after the encoder (see
    `encode`); every token axis downstream is pure patch tokens, width P.

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

    Args:
        encoder:        pre-built HF backbone; must expose .embeddings,
                        .encoder, .layernorm, .config. Downloads if None.
        preserve_ratio: fraction of the encoder's width (`d_enc`) the core
                        operates at: `core_dim = round(d_enc * preserve_ratio)`,
                        rounded to the nearest multiple of `core_heads` (attention
                        needs `core_dim % core_heads == 0`). This is what lets the
                        core be sized independently of whichever encoder is
                        loaded. `adapter` (`CNNAdapter`) reassembles the patch
                        tokens of a frame into their (sqrt(P), sqrt(P)) spatial
                        grid and reduces channels from `d_enc` to `core_dim`
                        with a couple of 3x3 convolutions, so the projection can
                        mix neighbouring patches instead of treating every token
                        independently. An EMA copy, `ema_adapter`, produces the
                        training target (see `ema_momentum`, `update_ema_adapter`)
                        so the loss compares adapted-space tensors instead of the
                        online adapter's raw encoder input.
        freeze_encoder: freeze the encoder and use it directly for targets
                        (detached). Cheapest and safest starting point.
                        `adapter` and `core` are never frozen by this --
                        they always train, since they're what has to learn
                        the encoder -> core projection.
        use_ema:        if the encoder is trainable, keep an EMA copy to
                        produce targets. Without one of freeze_encoder or
                        use_ema, gradients reach both sides of the loss and
                        the representation can collapse.
        context_mode:   'full'  -> attend every token of every earlier frame
                                   (L = N*P), causality via mask.
                        'state' -> attend only the GRU state after frame t-1
                                   (L = P). Strictly causal by construction,
                                   far cheaper, no mask needed. That state is
                                   tagged tau_{t-1}, i.e. "the summary as of
                                   the last observed frame".
        rope_time_periods: (fastest, slowest) period in the SAME UNITS as
                        `frame_times`. `RVMDataset` reports `frame_times` in
                        seconds, so the default is DINO-world's seconds-scale
                        range (1e-2, 1e2); rescale it if you feed some other unit.
        rope_space_periods: periods in normalized grid units. Range is 2.0
                        edge-to-edge, one patch is 2/grid, so ~(0.2, 4.0)
                        covers neighbour-level to whole-image structure.
        normalize_target: layer-norm each target token (over D) before the
                        loss. Without it, a handful of high-norm tokens --
                        CLS, and register-less DINOv2's high-norm artifact
                        patches -- dominate the mean over all tokens, since
                        F.smooth_l1_loss averages raw error across every
                        token indiscriminately. Per-token layer-norm puts
                        every token on the same scale before the loss sees
                        it; `pred` needs no matching treatment since
                        `repr_head` is free to learn output at that scale.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_name: str = "facebook/dinov2-with-registers-base",
        preserve_ratio: float = 0.2,
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
        rope_time_periods: Tuple[float, float] = (1e-2, 1e2),
        rope_space_periods: Tuple[float, float] = (0.2, 4.0),
        normalize_target: bool = False,
    ):
        super().__init__()
        assert context_mode in ("full", "state")

        if encoder is None:
            from transformers import AutoModel
            print(f'loading the vision encoder from {encoder_name}')
            encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder = encoder

        self.d_enc = encoder.config.hidden_size
        self.patch = encoder.config.patch_size
        # 0 for encoders without registers (e.g. plain dinov2-small).
        self.num_register_tokens = getattr(encoder.config, "num_register_tokens", 0)
        # round to a multiple of core_heads: attention requires core_dim % core_heads == 0,
        # and d_enc * preserve_ratio is not one in general (e.g. 768 * 0.2 = 153.6).
        self.core_dim = max(core_heads, round(self.d_enc * preserve_ratio / core_heads) * core_heads)
        self.preserve_ratio = preserve_ratio
        self.dec_dim = dec_dim
        self.freeze_encoder = freeze_encoder
        self.context_mode = context_mode
        self.checkpoint_core = checkpoint_core
        self.ema_momentum = ema_momentum
        self.loss_beta = loss_beta
        self.t_periods = rope_time_periods
        self.s_periods = rope_space_periods
        self.normalize_target = normalize_target

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

        # trainable CNN projection from encoder space into core space, applied to
        # every frame's patch tokens right before they reach `core` -- see
        # `core_dim`/`preserve_ratio` above.
        self.adapter = CNNAdapter(self.d_enc, self.core_dim)

        # EMA copy of `adapter`, updated by `update_ema_adapter` (never by gradient):
        # produces the training target in `forward`, so the loss compares the online
        # adapter's prediction against a slowly-moving version of itself rather than
        # a moving target it can trivially collapse onto.
        self.ema_adapter = copy.deepcopy(self.adapter)
        for p in self.ema_adapter.parameters():
            p.requires_grad_(False)
        self.ema_adapter.eval()

        core_mlp = core_mlp or 4 * self.core_dim
        self.core = GatedRecurrentCore(self.core_dim, core_heads, core_layers, core_mlp)

        self.decoder_embed = nn.Linear(self.core_dim, dec_dim)
        self.decoder = CrossAttentionTransformer(dec_dim, dec_heads, dec_layers, dec_mlp)
        self.repr_head = nn.Linear(dec_dim, self.core_dim)

        # the entire content of a query: one learnable vector, shared by every
        # patch of every target frame. Position enters only through rotation.
        self.query_token = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)

    @torch.no_grad()
    def update_ema_adapter(self):
        """Momentum update `ema_adapter = m*ema_adapter + (1-m)*adapter`. Call once
        per optimizer step (after the online `adapter` has been updated), never
        during forward -- this is not part of the computation graph."""
        m = self.ema_momentum
        for ema_p, p in zip(self.ema_adapter.parameters(), self.adapter.parameters()):
            ema_p.mul_(m).add_(p, alpha=1.0 - m)

    # encoder

    def _drop_cls(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[..., 1:, :]

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
        """(M, 3, H, W) -> (M, P, D) through the online encoder."""
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            h = self.encoder.encoder(self.encoder.embeddings(frames)).last_hidden_state
            h = self._drop_registers(h)
            return self._drop_cls(self.encoder.layernorm(h))

    def _grid(self, P: int, device) -> torch.Tensor:
        """(P, 2) spatial coords on [-1, +1].

        Rebuilt each call: it is a few hundred floats, and unlike a cached
        buffer it does not mutate module state, so it is compile- and
        DDP-friendly.
        """
        g = int(round(math.sqrt(P)))
        assert g * g == P, f"{P} patch tokens is not a square grid"
        lin = torch.linspace(-1.0, 1.0, g, device=device, dtype=torch.float32)
        ii, jj = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=-1)  # (P, 2)

    def _coords(self, tau: torch.Tensor, P: int) -> torch.Tensor:
        """tau (..., ) absolute times -> (..., P, 3) as (tau, i, j)."""
        device = tau.device
        grid = self._grid(P, device)                                  # (P, 2)
        sp = grid.expand(*tau.shape, P, 2)
        t = tau.to(torch.float32).unsqueeze(-1).unsqueeze(-1).expand(*tau.shape, P, 1)
        return torch.cat([t, sp], dim=-1)

    def _rope(self, coords: torch.Tensor, head_dim: int) -> RoPE:
        per = rope_periods(head_dim, self.t_periods, self.s_periods, coords.device)
        return axial_rope(coords, per)


    def _run_core(self, core_feats: torch.Tensor, frame_times: torch.Tensor,
                  state: torch.Tensor, t: int, n_tok: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step with the right time tags on x and on the state.
        `t` indexes position in the sampled clip; `frame_times[:, t]` is that
        frame's real timestamp in seconds, which is what actually enters RoPE.
        `core_feats` must already be in core space, i.e. passed through
        `self.adapter`."""
        hd = self.core.head_dim
        tau_now = frame_times[:, t]
        tau_prev = frame_times[:, max(t - 1, 0)]
        x_pos = self._rope(self._coords(tau_now, n_tok), hd)
        s_pos = self._rope(self._coords(tau_prev, n_tok), hd)
        if self.checkpoint_core and self.training:
            return checkpoint(self.core, core_feats[:, t], state, x_pos, s_pos,
                              use_reentrant=False)
        return self.core(core_feats[:, t], state, x_pos, s_pos)

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

        feats = self.encode(frames.reshape(B * N, *img))           # (B*N, P, D)
        feats = feats.view(B, N, *feats.shape[1:])                 # (B, N, P, D)
        n_tok = P = feats.shape[2]
        dtype = feats.dtype
        core_feats = self.adapter(feats)                           # (B, N, P, core_dim)

        if state is None:
            state = feats.new_zeros(B, n_tok, self.core_dim)
        memory = []
        for t in range(N):
            out, state = self._run_core(core_feats, frame_times, state, t, n_tok)
            memory.append(out)
        memory = torch.stack(memory, dim=1)                        # (B, N, n_tok, core_dim)

        hd = self.decoder.head_dim
        if self.context_mode == "full":
            L = N * n_tok
            kv = self.decoder_embed(memory.reshape(B, L, self.core_dim))  # bs, t, num_p, emb_d --> bs, t * num_p, emb_d
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
        pred = self.repr_head(decoded).view(B, Tt, P, self.core_dim)
        gap = frame_times[:, target_idx] - frame_times[:, target_idx - 1]

        # target lives in adapted (core) space, not raw encoder space, so it has to
        # go through `ema_adapter` rather than being compared against `feats` directly.
        with torch.no_grad():
            ema_target = self.ema_adapter(feats)                       # (B, N, P, core_dim)
        repr_loss = self.loss(pred=pred, target=ema_target, target_idx=target_idx)

        return {
            "pred": pred,                # (B, Tt, P, core_dim)
            "memory": memory,            # (B, N,  P, core_dim)
            "state": state,              # (B, P, core_dim)
            "gap": gap,                  # (B, Tt), reported only; RoPE carries it
            "grid": int(round(math.sqrt(P))),
            "repr_loss": repr_loss,
            "loss": repr_loss,
        }

    def _build_queries(self, n_rows: int, P: int, dtype) -> torch.Tensor:
        """(n_rows, P, Dd). Pure content -- no additive position at all."""
        return self.query_token.expand(n_rows, P, self.dec_dim).to(dtype)

    def loss(self, pred: torch.Tensor, target: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """target is `ema_adapter`'s output (B, N, P, core_dim); pick out the frames
        pred was built for."""
        tgt = target[:, target_idx, ...].detach()
        return F.smooth_l1_loss(pred, tgt, beta=self.loss_beta)


    @torch.no_grad()
    def rollout(
        self,
        context: torch.Tensor,                          # (B, C, 3, H, W)
        targets: Optional[torch.Tensor] = None,          # (B, S, 3, H, W), optional -- only encoded to report target_feat
        context_times: Optional[torch.Tensor] = None,    # (B, C) float
        target_times: Optional[torch.Tensor] = None,     # (B, S) float, absolute query timestamps
        state: Optional[torch.Tensor] = None,
    ) -> dict:
        """Autoregressive rollout in latent space. Returns pred of shape (B, S, n_tok, D).

        Each predicted frame is fed back through the recurrent core in place of
        encoder features, so this is the inference-time path the teacher-forced
        training objective is an approximation of. Expect the gap between the
        two to widen with horizon.

        Same call signature as `teacher_forcing_rollout_repr`: `target_times`
        supplies the absolute per-sample query timestamps directly (batched,
        so heterogeneous per-clip timing is fine). `targets` is optional and,
        if given, is only encoded to report `target_feat` for a direct diff
        against `teacher_forcing_rollout_repr`'s output -- it is never fed
        back into `core` here, which is what makes this the non-teacher-forced
        path.

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
        n_tok = P = feats.shape[2]
        dtype = feats.dtype
        core_feats = self.adapter(feats)

        if context_times is None:
            context_times = torch.arange(C, device=device, dtype=torch.float32)
            context_times = context_times.unsqueeze(0).expand(B, C)
        context_times = context_times.to(device=device, dtype=torch.float32)

        if target_times is None:
            assert targets is not None, "need target_times or targets to know rollout length"
            S = targets.shape[1]
            target_times = C + torch.arange(S, device=device, dtype=torch.float32)
            target_times = target_times.unsqueeze(0).expand(B, S)
        target_times = target_times.to(device=device, dtype=torch.float32)
        S = target_times.shape[1]

        target_feat = None
        if targets is not None:
            target_feat = self.encode(targets.reshape(B * S, *targets.shape[2:]))
            target_feat = target_feat.view(B, S, *target_feat.shape[1:])
            target_feat = self.ema_adapter(target_feat)   # adapted space, comparable to `pred`

        if state is None:
            state = feats.new_zeros(B, n_tok, self.core_dim)
        memory, times = [], []
        for t in range(C):
            out, state = self._run_core(core_feats, context_times, state, t, n_tok)
            memory.append(out)
            times.append(context_times[:, t])

        hd = self.decoder.head_dim
        preds = []
        for s in range(S):
            tau = target_times[:, s]
            if self.context_mode == "full":
                kv = torch.stack(memory, dim=1).reshape(B, -1, self.core_dim)
                kv_tau = torch.stack(times, dim=1)                        # (B, k)
            else:
                kv = memory[-1]
                kv_tau = times[-1].unsqueeze(1)                           # (B, 1)
            kv = self.decoder_embed(kv)
            kv_coords = self._coords(kv_tau, P).reshape(B, -1, 3)
            q_coords = self._coords(tau, P)

            q = self._build_queries(B, P, dtype)
            decoded = self.decoder(
                q, kv, None,
                q_pos=self._rope(q_coords, hd), kv_pos=self._rope(kv_coords, hd),
            )

            step = self.repr_head(decoded)     # already core-space
            preds.append(step)

            # feed the prediction back, with its own timestamp. Unlike a real
            # frame, `step` is already in core space (repr_head's output), so it
            # skips `adapter` and goes straight into `core`.
            prev_tau = times[-1]
            hdc = self.core.head_dim
            out, state = self.core(
                step, state,
                self._rope(self._coords(tau, P), hdc),
                self._rope(self._coords(prev_tau, P), hdc),
            )
            memory.append(out)
            times.append(tau)

        out = {
            "pred": torch.stack(preds, dim=1),
            "context_memory": torch.stack(memory, dim=1),
            "context_feat": feats,
            "state": state,
        }
        if target_feat is not None:
            out["target_feat"] = target_feat                 # (B, S, P, core_dim)
        return out

    @torch.no_grad()
    def teacher_forcing_rollout_repr(
        self,
        context: torch.Tensor,                          # (B, C, 3, H, W)
        targets: torch.Tensor,                           # (B, S, 3, H, W) ground-truth future frames
        context_times: Optional[torch.Tensor] = None,    # (B, C) float
        target_times: Optional[torch.Tensor] = None,     # (B, S) float
        state: Optional[torch.Tensor] = None,
    ) -> dict:
        """Representation-space teacher-forced rollout: like `rollout`, but the
        core's input at step s is the TRUE encoded target frame `targets[:, s]`
        (through `adapter`), not the model's own prediction fed back. This
        isolates decoder underfit from the core's autoregressive drift.
        Diff `out["pred"]` against `rollout`'s `pred` on the same
        clip to see how much of rollout's degradation is drift vs. decoder
        underfit.
        """
        self.eval()
        B, C = context.shape[:2]
        S = targets.shape[1]
        device = context.device

        ctx_feats = self.encode(context.reshape(B * C, *context.shape[2:]))
        ctx_feats = ctx_feats.view(B, C, *ctx_feats.shape[1:])
        tgt_feats = self.encode(targets.reshape(B * S, *targets.shape[2:]))
        tgt_feats = tgt_feats.view(B, S, *tgt_feats.shape[1:])
        n_tok = P = ctx_feats.shape[2]
        dtype = ctx_feats.dtype
        core_ctx_feats = self.adapter(ctx_feats)

        if context_times is None:
            context_times = torch.arange(C, device=device, dtype=torch.float32)
            context_times = context_times.unsqueeze(0).expand(B, C)
        context_times = context_times.to(device=device, dtype=torch.float32)

        if target_times is None:
            target_times = C + torch.arange(S, device=device, dtype=torch.float32)
            target_times = target_times.unsqueeze(0).expand(B, S)
        target_times = target_times.to(device=device, dtype=torch.float32)

        if state is None:
            state = ctx_feats.new_zeros(B, n_tok, self.core_dim)
        memory, times = [], []
        for t in range(C):
            out, state = self._run_core(core_ctx_feats, context_times, state, t, n_tok)
            memory.append(out)
            times.append(context_times[:, t])

        hd, hdc = self.decoder.head_dim, self.core.head_dim
        preds = []
        for s in range(S):
            tau = target_times[:, s]

            if self.context_mode == "full":
                kv = torch.stack(memory, dim=1).reshape(B, -1, self.core_dim)
                kv_tau = torch.stack(times, dim=1)                 # (B, k)
            else:
                kv = memory[-1]
                kv_tau = times[-1].unsqueeze(1)                    # (B, 1)
            kv = self.decoder_embed(kv)
            kv_coords = self._coords(kv_tau, P).reshape(B, -1, 3)
            q_coords = self._coords(tau, P)

            q = self._build_queries(B, P, dtype)
            decoded = self.decoder(
                q, kv, None,
                q_pos=self._rope(q_coords, hd), kv_pos=self._rope(kv_coords, hd),
            )

            pred = self.repr_head(decoded)      # (B, P, core_dim)
            preds.append(pred)

            # teacher forcing: core sees the TRUE target frame's encoding, not
            # its own prediction fed back -- through the adapter, same as
            # every other path into `core`.
            prev_tau = times[-1]
            out, state = self.core(
                self.adapter(tgt_feats[:, s]), state,
                self._rope(self._coords(tau, P), hdc),
                self._rope(self._coords(prev_tau, P), hdc),
            )
            memory.append(out)
            times.append(tau)

        return {
            "pred": torch.stack(preds, dim=1),              # (B, S, P, core_dim)
            "target_feat": self.ema_adapter(tgt_feats),      # (B, S, P, core_dim), adapted space -- comparable to `pred`
            "context_memory": torch.stack(memory, dim=1),    # (B, C+S, P, core_dim) core outputs, context + teacher-forced steps
            "context_feat": ctx_feats,                        # (B, C, P, D) raw encoder features for the context frames
            "state": state,
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
        B, P = tokens.shape[0], tokens.shape[1]
        if state is None:
            state = tokens.new_zeros(B, P, self.core_dim)
        if t_now is None:
            t_now = tokens.new_zeros(B)
        if t_prev is None:
            t_prev = t_now
        hd = self.core.head_dim
        return self.core(
            self.adapter(tokens), state,
            self._rope(self._coords(t_now.to(tokens.device), P), hd),
            self._rope(self._coords(t_prev.to(tokens.device), P), hd),
        )
