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
# from .utils.rope import RoPE, axial_rope, rope_periods

def rope_periods_time(head_dim: int, t_range: Tuple[float, float], device) -> torch.Tensor:
    """Angular periods for one head, time-only RoPE: (R,) with R = head_dim // 2.

    1-axial analogue of `rope_periods`: with no spatial axes to share the
    budget with, every rotating channel pair gets a time frequency.
    """
    per = head_dim // 2
    assert per > 0, f"head_dim {head_dim} too small for RoPE"
    return torch.logspace(math.log10(t_range[0]), math.log10(t_range[1]), per, device=device, dtype=torch.float32)


def time_rope(tau: torch.Tensor, periods: torch.Tensor) -> RoPE:
    """tau (..., N) absolute times -> (cos, sin), each (..., N, R). 1-axial analogue of `axial_rope`."""
    ang = 2.0 * math.pi * tau.to(torch.float32).unsqueeze(-1) / periods
    return ang.cos(), ang.sin()


class RecurrentWorldModelCLS(nn.Module):
    """CLS-token-only recurrent world model. See module note above.

    Args mirror `RecurrentWorldModel` where they still apply; patch/grid-only
    concepts (drop_cls, rope_space_periods, pixel_recon, objective,
    normalize_target) have no counterpart here and are dropped rather than
    carried over unused.

    Args:
        encoder:        pre-built HF backbone; must expose .embeddings,
                        .encoder, .layernorm, .config. Downloads if None.
        freeze_encoder: freeze the encoder and use it directly for targets
                        (detached).
        use_ema:        if the encoder is trainable, keep an EMA copy to
                        produce targets.
        context_mode:   'full'  -> attend every earlier frame's CLS token
                                   (L = N), causality via mask.
                        'state' -> attend only the GRU state after frame t-1
                                   (L = 1). Strictly causal by construction.
        rope_time_periods: (fastest, slowest) period in the SAME UNITS as
                        `frame_times`.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_name: str = "facebook/dinov2-with-registers-large",
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
    ):
        super().__init__()
        assert context_mode in ("full", "state")

        if encoder is None:
            from transformers import AutoModel
            print(f"loading the vision encoder from {encoder_name}")
            encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder = encoder

        self.d_enc = encoder.config.hidden_size
        self.dec_dim = dec_dim
        self.freeze_encoder = freeze_encoder
        self.context_mode = context_mode
        self.checkpoint_core = checkpoint_core
        self.ema_momentum = ema_momentum
        self.loss_beta = loss_beta
        self.t_periods = rope_time_periods

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
        # target frame -- there is only one token per frame now, so there is
        # no separate cls_query/query_token split. Position enters only
        # through rotation.
        self.query_token = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)

    def encode_cls(self, frames: torch.Tensor) -> torch.Tensor:
        """(M, 3, H, W) -> (M, D), the [CLS] token only, through the online encoder."""
        ctx = torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()
        with ctx:
            h = self.encoder.encoder(self.encoder.embeddings(frames)).last_hidden_state
            return self.encoder.layernorm(h)[:, 0, :]

    def _rope_cls(self, tau: torch.Tensor, head_dim: int) -> RoPE:
        periods = rope_periods_time(head_dim, self.t_periods, tau.device)
        return time_rope(tau, periods)

    def _run_core(self, feats: torch.Tensor, frame_times: torch.Tensor,
                  state: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step with the right time tags on x and on the state.
        feats/state are (B, D); the core itself needs a token axis, so it is
        added here (length 1) and squeezed back off on the way out."""
        hd = self.core.head_dim
        tau_now = frame_times[:, t: t + 1]
        tau_prev = frame_times[:, max(t - 1, 0): max(t - 1, 0) + 1]
        x_pos = self._rope_cls(tau_now, hd)
        s_pos = self._rope_cls(tau_prev, hd)
        x = feats[:, t: t + 1, :]
        s = state.unsqueeze(1)
        if self.checkpoint_core and self.training:
            out, new_state = checkpoint(self.core, x, s, x_pos, s_pos, use_reentrant=False)
        else:
            out, new_state = self.core(x, s, x_pos, s_pos)
        return out.squeeze(1), new_state.squeeze(1)

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

        if frame_times is None:
            frame_times = torch.arange(N, device=device, dtype=torch.float32)
            frame_times = frame_times.unsqueeze(0).expand(B, N)
        frame_times = frame_times.to(device=device, dtype=torch.float32)

        feats = self.encode_cls(frames.reshape(B * N, *img))        # (B*N, D)
        feats = feats.view(B, N, self.d_enc)                        # (B, N, D)
        dtype = feats.dtype

        if state is None:
            state = feats.new_zeros(B, self.d_enc)
        memory = []
        for t in range(N):
            out, state = self._run_core(feats, frame_times, state, t)
            memory.append(out)
        memory = torch.stack(memory, dim=1)                         # (B, N, D)

        hd = self.decoder.head_dim
        if self.context_mode == "full":
            kv = self.decoder_embed(memory)                                     # (B, N, Dd)
            kv = kv.unsqueeze(1).expand(B, Tt, N, self.dec_dim).reshape(B * Tt, N, self.dec_dim)
            kv_tau = frame_times.unsqueeze(1).expand(B, Tt, N).reshape(B * Tt, N)

            # memory[:, t] summarizes everything up to and including frame t.
            frame_idx = torch.arange(N, device=device)
            allowed = frame_idx.view(1, N) < target_idx.view(Tt, 1)              # (Tt, N)
            attn_mask = (
                allowed.view(1, Tt, 1, 1, N)
                .expand(B, Tt, 1, 1, N)
                .reshape(B * Tt, 1, 1, N)
            )
        else:  # 'state': the GRU state after frame t-1 already is the history
            kv = self.decoder_embed(memory[:, target_idx - 1]).reshape(B * Tt, 1, self.dec_dim)
            kv_tau = frame_times[:, target_idx - 1].reshape(B * Tt, 1)
            attn_mask = None

        q_tau = frame_times[:, target_idx].reshape(B * Tt, 1)
        queries = self.query_token.expand(B * Tt, 1, self.dec_dim).to(dtype)

        decoded = self.decoder(
            queries, kv, attn_mask,
            q_pos=self._rope_cls(q_tau, hd), kv_pos=self._rope_cls(kv_tau, hd),
        )
        pred = self.repr_head(decoded).view(B, Tt, self.d_enc)      # (B, Tt, D)

        gap = frame_times[:, target_idx] - frame_times[:, target_idx - 1]
        repr_loss = self.loss(pred=pred, target=feats, target_idx=target_idx)

        return {
            "pred": pred,                # (B, Tt, D)
            "memory": memory,            # (B, N, D)
            "state": state,              # (B, D)
            "gap": gap,                  # (B, Tt), reported only; RoPE carries it
            "repr_loss": repr_loss,
            "loss": repr_loss,
        }

    def loss(self, pred: torch.Tensor, target: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        """target is `feats` (B, N, D); pick out the frames pred was built for."""
        tgt = target[:, target_idx, :].detach()
        return F.smooth_l1_loss(pred, tgt, beta=self.loss_beta)
