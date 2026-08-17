from __future__ import annotations

import contextlib
import copy
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of arbitrary (possibly fractional) positions.

    Args:
        positions: (M,) float tensor. Patch indices, time gaps, whatever.
        dim:       embedding width.
    Returns:
        (M, dim)
    """
    j = torch.arange(dim, device=positions.device, dtype=torch.float32)
    denom = torch.pow(10000.0, 2 * torch.div(j, 2, rounding_mode="floor") / dim)
    a = positions.float().unsqueeze(1) / denom
    out = torch.empty_like(a)
    out[:, 0::2] = torch.sin(a[:, 0::2])
    out[:, 1::2] = torch.cos(a[:, 1::2])
    return out


def sample_frame_times(
    batch: int,
    n_frames: int,
    gap_min: float = 1.0,
    gap_max: float = 12.0,
    device=None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Cumulative-sum of uniform gaps, as in DINO-world's variable-FPS sampling.

    Returns (batch, n_frames) float "times" starting at 0. Use these to pick the
    nearest real frame from each video, and pass the same tensor to `forward`.
    """
    gaps = torch.rand(batch, n_frames - 1, device=device, generator=generator)
    gaps = gap_min + gaps * (gap_max - gap_min)
    zero = gaps.new_zeros(batch, 1)
    return torch.cat([zero, gaps.cumsum(dim=1)], dim=1)



class MultiHeadAttention(nn.Module):
    """Self- or cross-attention with optional additive/boolean mask."""

    def __init__(self, d_model: int, num_heads: int, d_attn: Optional[int] = None):
        super().__init__()
        d_attn = d_attn or d_model
        assert d_attn % num_heads == 0, "d_attn must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_attn // num_heads

        self.q = nn.Linear(d_model, d_attn)
        self.k = nn.Linear(d_model, d_attn)
        self.v = nn.Linear(d_model, d_attn)
        self.out = nn.Linear(d_attn, d_model)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, N, _ = t.shape
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kv = x if kv is None else kv
        q, k, v = self._split(self.q(x)), self._split(self.k(kv)), self._split(self.v(kv))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out(o)


class CrossAttentionBlock(nn.Module):
    """Pre-norm: cross-attn -> MLP -> self-attn.

    The mask applies only to cross-attention. Self-attention mixes a single
    target frame's own query tokens, which is always legal.
    """

    def __init__(self, d_model: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.ca_norm = nn.LayerNorm(d_model)
        self.ca = MultiHeadAttention(d_model, num_heads)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, d_model)
        )
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, num_heads)

    def forward(self, x, kv, attn_mask=None):
        x = x + self.ca(self.ca_norm(x), kv, attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        x = x + self.sa(self.sa_norm(x))
        return x


class CrossAttentionTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, mlp_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, kv, attn_mask=None):
        for blk in self.blocks:
            x = blk(x, kv, attn_mask)
        return self.out_norm(x)


class GatedRecurrentCore(nn.Module):
    """GRU over a whole token set, cross-attention transformer as candidate.

        z  = sigmoid(W_z x + U_z h)
        r  = sigmoid(W_r x + U_r h)
        h~ = XAttn(q=x, kv=r * LN(h))
        h  <- (1 - z) * h + z * h~
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, mlp_dim: int):
        super().__init__()
        self.input_update = nn.Linear(d_model, d_model, bias=False)
        self.state_update = nn.Linear(d_model, d_model, bias=False)
        self.input_reset = nn.Linear(d_model, d_model, bias=False)
        self.state_reset = nn.Linear(d_model, d_model, bias=False)
        self.state_norm = nn.LayerNorm(d_model, eps=1e-4, bias=False)
        self.transformer = CrossAttentionTransformer(d_model, num_heads, num_layers, mlp_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        z = torch.sigmoid(self.input_update(x) + self.state_update(state))
        r = torch.sigmoid(self.input_reset(x) + self.state_reset(state))
        h = self.transformer(x, r * self.state_norm(state))
        out = (1.0 - z) * state + z * h
        return out, out



class RecurrentWorldModel(nn.Module):
    """Shape symbols used throughout:

        B   batch                 N   frames per clip
        P   patches per frame     D   encoder width      Dd  decoder width
        Tt  target frames         L   = N * (1 + P)  flattened memory length

    Token axis is always [CLS, patch_0 .. patch_{P-1}], so width is 1 + P.

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
                                   far cheaper, no mask needed.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_name: str = "facebook/dinov2-small",
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
    ):
        super().__init__()
        assert context_mode in ("full", "state")

        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder = encoder

        self.d_enc = encoder.config.hidden_size
        self.patch = encoder.config.patch_size
        self.dec_dim = dec_dim
        self.freeze_encoder = freeze_encoder
        self.context_mode = context_mode
        self.checkpoint_core = checkpoint_core
        self.ema_momentum = ema_momentum
        self.loss_beta = loss_beta

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

        # the entire content of a query: one learnable vector, shared by every
        # patch of every target frame. Position and time are added on top.
        self.query_token = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)
        self.cls_query = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)

        self.register_buffer("_posenc", torch.zeros(0), persistent=False)

    # ----------------------------- encoder utils ---------------------------- #

    def _embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(M, 3, H, W) -> (M, 1+P, D). CLS prepended, ViT pos-emb already added."""
        return self.encoder.embeddings(pixel_values)

    def _blocks(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.encoder.encoder(tokens).last_hidden_state
        return self.encoder.layernorm(h)

    def _encoder_ctx(self):
        # nullcontext, NOT enable_grad: enable_grad would build a graph even
        # when the caller wrapped the model in torch.no_grad().
        return torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(M, 3, H, W) -> (M, 1+P, D) through the online encoder."""
        with self._encoder_ctx():
            return self._blocks(self._embed(frames))

    @torch.no_grad()
    def encode_target(self, frames: torch.Tensor) -> torch.Tensor:
        """Detached features used as the regression target."""
        if self.use_ema:
            h = self.target_encoder.encoder(
                self.target_encoder.embeddings(frames)
            ).last_hidden_state
            return self.target_encoder.layernorm(h)
        return self._blocks(self._embed(frames)).detach()

    @torch.no_grad()
    def update_ema(self, momentum: Optional[float] = None) -> None:
        """Call once per optimizer step when use_ema=True."""
        if not self.use_ema:
            return
        m = self.ema_momentum if momentum is None else momentum
        for pt, ps in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bt, bs in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            bt.copy_(bs)

    # ------------------------------ query build ----------------------------- #

    def _spatial_posenc(self, n: int, device, dtype) -> torch.Tensor:
        if self._posenc.shape[0] != n or self._posenc.device != device:
            pos = torch.arange(n, device=device, dtype=torch.float32)
            self._posenc = sincos_1d(pos, self.dec_dim)
        return self._posenc.to(dtype)

    def _build_queries(self, n_rows: int, P: int, gap: torch.Tensor,
                       device, dtype) -> torch.Tensor:
        """(n_rows, 1+P, Dd). `gap` is (n_rows,) float, time to previous frame."""
        q = self.query_token.expand(n_rows, P, self.dec_dim).to(dtype)
        q = q + self._spatial_posenc(P, device, dtype).unsqueeze(0)
        time = sincos_1d(gap.to(device), self.dec_dim).to(dtype).unsqueeze(1)
        q = q + time
        cls = self.cls_query.expand(n_rows, 1, self.dec_dim).to(dtype) + time
        return torch.cat([cls, q], dim=1)


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
        target_idx = target_idx.to(device).long()
        Tt = int(target_idx.numel())
        assert int(target_idx.min()) >= 1, "frame 0 has no history; it cannot be a target"
        assert int(target_idx.max()) < N, "target_idx out of range"

        if frame_times is None:                       # contiguous fallback
            frame_times = torch.arange(N, device=device, dtype=torch.float32)
            frame_times = frame_times.unsqueeze(0).expand(B, N)

        feats = self.encode(frames.reshape(B * N, *img))          # (B*N, 1+P, D)
        feats = feats.view(B, N, *feats.shape[1:])                # (B, N, 1+P, D)
        P = feats.shape[2] - 1
        dtype = feats.dtype

        if state is None:
            state = feats.new_zeros(B, 1 + P, self.d_enc)
        memory = []
        for t in range(N):
            frame = feats[:, t]
            if self.checkpoint_core and self.training:
                out, state = checkpoint(self.core, frame, state, use_reentrant=False)
            else:
                out, state = self.core(frame, state)
            memory.append(out)
        memory = torch.stack(memory, dim=1)                       # (B, N, 1+P, D)

        if self.context_mode == "full":
            L = N * (1 + P)
            kv = self.decoder_embed(memory.reshape(B, L, self.d_enc))       # (B, L, Dd)
            kv = kv.unsqueeze(1).expand(B, Tt, L, self.dec_dim)
            kv = kv.reshape(B * Tt, L, self.dec_dim)

            frame_of_tok = torch.arange(N, device=device).repeat_interleave(1 + P)
            allowed = frame_of_tok.view(1, L) < target_idx.view(Tt, 1)      # (Tt, L)
            attn_mask = (
                allowed.view(1, Tt, 1, 1, L)
                .expand(B, Tt, 1, 1, L)
                .reshape(B * Tt, 1, 1, L)
            )
        else:  # 'state': the GRU state after frame t-1 already is the history
            kv = memory[:, target_idx - 1]                                  # (B,Tt,1+P,D)
            kv = self.decoder_embed(kv).reshape(B * Tt, 1 + P, self.dec_dim)
            attn_mask = None

        gap = (frame_times[:, target_idx] - frame_times[:, target_idx - 1])  # (B, Tt)
        queries = self._build_queries(B * Tt, P, gap.reshape(-1), device, dtype)

        decoded = self.decoder(queries, kv, attn_mask)             # (B*Tt, 1+P, Dd)
        pred = self.repr_head(decoded).view(B, Tt, 1 + P, self.d_enc)

        target = self.encode_target(
            frames[:, target_idx].reshape(B * Tt, *img)
        ).view(B, Tt, 1 + P, self.d_enc)

        return {
            "pred": pred,                # (B, Tt, 1+P, D)
            "target": target,            # (B, Tt, 1+P, D)  detached
            "memory": memory,            # (B, N,  1+P, D)
            "state": state,              # (B, 1+P, D)
            "gap": gap,                  # (B, Tt)
            "grid": int(round(math.sqrt(P))),
        }

    def loss(self, out: dict, include_cls: bool = False) -> torch.Tensor:
        pred, target = out["pred"], out["target"]
        if not include_cls:
            pred, target = pred[:, :, 1:], target[:, :, 1:]
        return F.smooth_l1_loss(pred, target.detach(), beta=self.loss_beta)


    @torch.no_grad()
    def rollout(
        self,
        context: torch.Tensor,           # (B, C, 3, H, W)
        gaps: Sequence[float] | torch.Tensor,   # per-step time gaps to predict
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout in latent space. Returns (B, S, 1+P, D).

        Each predicted frame is fed back through the recurrent core in place of
        encoder features, so this is the inference-time path the teacher-forced
        training objective is an approximation of. Expect the gap between the
        two to widen with horizon.
        """
        self.eval()
        B, C = context.shape[:2]
        device = context.device
        feats = self.encode(context.reshape(B * C, *context.shape[2:]))
        feats = feats.view(B, C, *feats.shape[1:])
        P = feats.shape[2] - 1
        dtype = feats.dtype

        if state is None:
            state = feats.new_zeros(B, 1 + P, self.d_enc)
        memory = []
        for t in range(C):
            out, state = self.core(feats[:, t], state)
            memory.append(out)

        if not torch.is_tensor(gaps):
            gaps = torch.tensor(list(gaps), device=device, dtype=torch.float32)
        gaps = gaps.to(device).float()

        preds = []
        for s in range(gaps.numel()):
            if self.context_mode == "full":
                mem = torch.stack(memory, dim=1)                  # (B, k, 1+P, D)
                kv = mem.reshape(B, -1, self.d_enc)
            else:
                kv = memory[-1]
            kv = self.decoder_embed(kv)

            gap_s = gaps[s].expand(B)
            q = self._build_queries(B, P, gap_s, device, dtype)
            step = self.repr_head(self.decoder(q, kv, None))      # (B, 1+P, D)
            preds.append(step)

            out, state = self.core(step, state)                   # feed prediction back
            memory.append(out)

        return torch.stack(preds, dim=1)

    @torch.no_grad()
    def step(self, frame: torch.Tensor, state: Optional[torch.Tensor] = None):
        """Streaming: one frame in, (features, new_state) out."""
        tokens = self.encode(frame)
        if state is None:
            state = tokens.new_zeros(tokens.shape)
        return self.core(tokens, state)



if __name__ == "__main__":
    torch.manual_seed(0)

    model = RecurrentWorldModel(
        encoder_name="facebook/dinov2-small",
        freeze_encoder=True,
        context_mode="full",
        dec_layers=2,
    )

    B, N = 2, 6
    frames = torch.randn(B, N, 3, 224, 224)
    times = sample_frame_times(B, N, gap_min=1.0, gap_max=12.0)
    target_idx = torch.tensor([3, 4, 5])

    out = model(frames, target_idx, times)
    print({k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
    print("loss:", model.loss(out).item())

    roll = model.rollout(frames[:, :3], gaps=[4.0, 4.0, 8.0])
    print("rollout:", tuple(roll.shape))

    # causality check: perturbing a frame at or after the earliest target must
    # not change that target's prediction; perturbing an earlier one must.
    m2 = RecurrentWorldModel(
        encoder_name="facebook/dinov2-small",
        freeze_encoder=True, context_mode="full", dec_layers=2,
    ).eval()
    f2 = frames.clone()
    f2[:, 4] = torch.randn_like(f2[:, 4])          # frame 4 >= target 3
    a = m2(frames, torch.tensor([3]), times)["pred"]
    b = m2(f2, torch.tensor([3]), times)["pred"]
    print("no leak from frame>=target:", torch.allclose(a, b, atol=1e-5))

    f3 = frames.clone()
    f3[:, 1] = torch.randn_like(f3[:, 1])          # frame 1 < target 3
    c = m2(f3, torch.tensor([3]), times)["pred"]
    print("does depend on history:", not torch.allclose(a, c, atol=1e-5))