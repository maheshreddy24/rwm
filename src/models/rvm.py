

import contextlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sincos_pos_embed(n_positions: int, dim: int, device=None, dtype=torch.float32):
    """1-D sin-cos table over flattened patch indices.

    Matches `get_mae_sinusoid_encoding_table` in the JAX code: the frequencies run
    over the *flattened* index, not over (row, col) separately, so this is a 1-D
    schedule reshaped to a grid rather than a true 2-D sin-cos embedding.
    """
    pos = torch.arange(n_positions, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(dim, dtype=torch.float64)
    denom = torch.pow(10000.0, 2 * torch.div(j, 2, rounding_mode="floor") / dim)
    table = pos / denom
    table[:, 0::2] = torch.sin(table[:, 0::2])
    table[:, 1::2] = torch.cos(table[:, 1::2])
    return table.to(device=device, dtype=dtype)


def random_masking(x: torch.Tensor, mask_ratio: float, generator=None):
    """Drop a random `mask_ratio` fraction of tokens.

    Args:
        x: (B, N, D)
    Returns:
        visible:     (B, n_keep, D)
        ids_restore: (B, N) indices that undo the shuffle
        mask:        (B, N) 1 = dropped, 0 = kept, in original order
    """
    B, N, D = x.shape
    n_keep = int(N * (1.0 - mask_ratio))
    noise = torch.rand(B, N, device=x.device, generator=generator)

    ids_shuffle = noise.argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :n_keep]

    visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device, dtype=x.dtype)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return visible, ids_restore, mask


def patchify(imgs: torch.Tensor, patch: int) -> torch.Tensor:
    """(B, 3, H, W) -> (B, N, patch*patch*3), row-major, matching DINOv2 patch order."""
    B, C, H, W = imgs.shape
    h, w = H // patch, W // patch
    x = imgs.reshape(B, C, h, patch, w, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)              # B h w p p C
    return x.reshape(B, h * w, patch * patch * C)


def unpatchify(x: torch.Tensor, patch: int, grid: int, C: int = 3) -> torch.Tensor:
    """(B, N, patch*patch*3) -> (B, 3, H, W)."""
    B = x.shape[0]
    x = x.reshape(B, grid, grid, patch, patch, C)
    x = x.permute(0, 5, 1, 3, 2, 4)              # B C h p w p
    return x.reshape(B, C, grid * patch, grid * patch)


class MultiHeadAttention(nn.Module):
    """Self- or cross-attention. Scaling is 1/sqrt(head_dim), as in the JAX code."""

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

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        kv = x if kv is None else kv
        q, k, v = self._split(self.q(x)), self._split(self.k(kv)), self._split(self.v(kv))
        o = F.scaled_dot_product_attention(q, k, v)          # scales by 1/sqrt(head_dim)
        o = o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out(o)


class CrossAttentionBlock(nn.Module):
    """Pre-norm block, cross-attn -> MLP -> self-attn (the JAX block's ordering)."""

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

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        x = x + self.ca(self.ca_norm(x), kv)
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

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, kv)
        return self.out_norm(x)


class GatedRecurrentCore(nn.Module):
    """GRU over a whole token set, with a cross-attention transformer as candidate.

        z = sigmoid(W_z x + U_z h)
        r = sigmoid(W_r x + U_r h)
        h~ = XAttn(q=x, kv=r * LN(h))
        h <- (1 - z) * h + z * h~
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, mlp_dim: int):
        super().__init__()
        self.input_update = nn.Linear(d_model, d_model, bias=False)
        self.state_update = nn.Linear(d_model, d_model, bias=False)
        self.input_reset = nn.Linear(d_model, d_model, bias=False)
        self.state_reset = nn.Linear(d_model, d_model, bias=False)
        # scale-only LayerNorm, eps=1e-4, mirroring the JAX config
        self.state_norm = nn.LayerNorm(d_model, eps=1e-4, bias=False)
        self.transformer = CrossAttentionTransformer(d_model, num_heads, num_layers, mlp_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        z = torch.sigmoid(self.input_update(x) + self.state_update(state))
        r = torch.sigmoid(self.input_reset(x) + self.state_reset(state))
        h = self.transformer(x, r * self.state_norm(state))
        out = (1.0 - z) * state + z * h
        return out, out


class RVM(nn.Module):
    """Recurrent video MAE with a DINOv2 encoder.

    Args:
        encoder_name: HF id, e.g. "facebook/dinov2-small" (384-dim, patch 14).
        encoder: pre-built backbone; skips the HF download. Must expose
            `.embeddings`, `.encoder`, `.layernorm`, `.config`.
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov2-small",
        encoder: Optional[nn.Module] = None,
        core_layers: int = 4,          # paper: 4 at every model scale (Table 5)
        core_heads: int = 8,           # paper: 8 / 12 / 16 / 16 for S / B / L / H
        core_mlp: Optional[int] = None,  # paper: MLP ratio 4.0
        dec_dim: int = 512,            # paper: decoder fixed across scales (Table 6)
        dec_layers: int = 8,
        dec_heads: int = 16,
        dec_mlp: int = 2048,
        mask_ratio: float = 0.95,
        max_delta: int = 64,
        freeze_encoder: bool = False,
        checkpoint_core: bool = False,
    ):
        super().__init__()
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder = encoder

        self.d_enc = encoder.config.hidden_size
        self.patch = encoder.config.patch_size
        self.dec_dim = dec_dim
        self.mask_ratio = mask_ratio
        self.freeze_encoder = freeze_encoder
        self.checkpoint_core = checkpoint_core

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        core_mlp = core_mlp or 4 * self.d_enc
        self.core = GatedRecurrentCore(self.d_enc, core_heads, core_layers, core_mlp)

        self.decoder_embed = nn.Linear(self.d_enc, dec_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)
        # equivalent to Linear(one_hot(delta, 64)) in the JAX code, minus the bias
        self.delta_embed = nn.Embedding(max_delta, dec_dim)
        nn.init.normal_(self.delta_embed.weight, std=0.02)
        self.max_delta = max_delta

        self.decoder = CrossAttentionTransformer(dec_dim, dec_heads, dec_layers, dec_mlp)
        self.head = nn.Linear(dec_dim, self.patch * self.patch * 3)

        self.register_buffer("_posenc", torch.zeros(0), persistent=False)


    def _embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, 1 + N, D), CLS prepended and pos-emb already added."""
        return self.encoder.embeddings(pixel_values)

    def _blocks(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the transformer stack on an arbitrary token set (no position logic)."""
        h = self.encoder.encoder(tokens).last_hidden_state
        return self.encoder.layernorm(h)

    def _encoder_ctx(self):
        # nullcontext, NOT enable_grad: enable_grad would force a graph to be built
        # even when the caller wrapped the whole model in torch.no_grad().
        return torch.no_grad() if self.freeze_encoder else contextlib.nullcontext()

    def _decoder_posenc(self, n: int, device, dtype) -> torch.Tensor:
        if self._posenc.shape[0] != n or self._posenc.device != device:
            self._posenc = sincos_pos_embed(n, self.dec_dim, device, torch.float32)
        return self._posenc.to(dtype)


    @torch.no_grad()
    def encode_frame(self, frame: torch.Tensor, state: Optional[torch.Tensor] = None):
        """One frame in, updated memory out. `frame`: (B, 3, H, W)."""
        tokens = self._blocks(self._embed(frame))
        if state is None:
            state = tokens.new_zeros(tokens.shape)
        features, state = self.core(tokens, state)
        return features, state


    def forward(
        self,
        source: torch.Tensor,          # (B, Ts, 3, H, W)
        target: torch.Tensor,          # (B, Tt, 3, H, W)
        target_deltas: torch.Tensor,   # (B, Tt)  int64, frame gap to last source frame
        state: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """Symbols used in the shape comments below:

            B   batch          Ts  source frames     Tt  target frames
            N   patches/frame  D   encoder width     Dd  decoder width
            M   visible target patches = floor((1 - mask_ratio) * N)
            L   memory length  = Ts * (1 + N)
        """
        B, Ts = source.shape[:2]
        Tt = target.shape[1]
        img = source.shape[2:]                                         # (3, H, W)

        # Fold time into the batch axis so frames are encoded independently:
        # attention contracts over the token axis, never over dim 0.
        with self._encoder_ctx():
            src = self._embed(source.reshape(B * Ts, *img))             # (B*Ts, 1+N, D)
            src = self._blocks(src)                                     # (B*Ts, 1+N, D)
        src = src.view(B, Ts, *src.shape[1:])                           # (B, Ts, 1+N, D)

        if state is None:
            state = src.new_zeros(B, src.shape[2], self.d_enc)          # (B, 1+N, D)  s_0 = 0
        else:
            assert state.shape == (B, src.shape[2], self.d_enc), (
                f"state {tuple(state.shape)} != {(B, src.shape[2], self.d_enc)}"
            )

        # Unroll the GRU core over time. This is the only temporal mixing.
        memory = []
        for t in range(Ts):
            frame = src[:, t]                                           # (B, 1+N, D)
            if self.checkpoint_core and self.training:
                out, state = checkpoint(self.core, frame, state, use_reentrant=False)
            else:
                out, state = self.core(frame, state)                    # o_t, s_t: (B, 1+N, D)
            memory.append(out)
        memory = torch.stack(memory, dim=1)                             # (B, Ts, 1+N, D)

        with self._encoder_ctx():
            emb = self._embed(target.reshape(B * Tt, *target.shape[2:]))  # (B*Tt, 1+N, D)
        cls_emb, patch_emb = emb[:, :1], emb[:, 1:]                     # (B*Tt, 1, D), (B*Tt, N, D)

        N = patch_emb.shape[1]
        grid = int(round(math.sqrt(N)))
        assert grid * grid == N, f"non-square token grid: {N} patches"

        visible, ids_restore, mask = random_masking(
            patch_emb, self.mask_ratio, generator
        )                                                               # (B*Tt, M, D), (B*Tt, N), (B*Tt, N)

        with self._encoder_ctx():
            tgt = self._blocks(torch.cat([cls_emb, visible], dim=1))    # (B*Tt, 1+M, D)

        #! this self.decoder_embed is a linear layer to reduce the dim of the embeddings
        tgt = self.decoder_embed(tgt)                                   # (B*Tt, 1+M, Dd)
        tgt_cls, tgt_vis = tgt[:, :1], tgt[:, 1:]                       # (B*Tt, 1, Dd), (B*Tt, M, Dd)

        mtok = self.mask_token.expand(B * Tt, N - tgt_vis.shape[1], -1)  # (B*Tt, N-M, Dd)
        full = torch.cat([tgt_vis, mtok], dim=1)                        # (B*Tt, N, Dd)  shuffled order
        full = torch.gather(                                            # undo the shuffle
            full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.dec_dim)
        )                                                             
          # (B*Tt, N, Dd)  grid order

        
        # ! here the target_deltas is the index into future timestamp, to be reconstructed.
        deltas = target_deltas.reshape(-1).clamp(0, self.max_delta - 1)  # (B*Tt,)
        full = full + self.delta_embed(deltas).unsqueeze(1)             # (B*Tt, 1, Dd) broadcast over N
        full = full + self._decoder_posenc(N, full.device, full.dtype)  # (N, Dd)       broadcast over B*Tt
        queries = torch.cat([tgt_cls, full], dim=1)                     # (B*Tt, 1+N, Dd)

        # the (B, Ts, 1+N, D) --> (bs, L, D). 
        # Here time IS flattened into the token axis on purpose: every target token
        # must be able to attend to every source token from every source frame.
        kv = memory.reshape(B, Ts * memory.shape[2], self.d_enc)        # (B, L, D)
        kv = self.decoder_embed(kv)                                     # (B, L, Dd)
        kv = kv.unsqueeze(1).expand(B, Tt, -1, -1)                      # (B, Tt, L, Dd)  view, no copy
        kv = kv.reshape(B * Tt, -1, self.dec_dim)                       # (B*Tt, L, Dd)   copies here

        decoded = self.decoder(queries, kv)                             # (B*Tt, 1+N, Dd)
        pred = self.head(decoded[:, 1:])                                # (B*Tt, N, p*p*3)

        return {
            "pred": pred.view(B, Tt, N, -1),                            # (B, Tt, N, p*p*3)
            "mask": mask.view(B, Tt, N),                                # (B, Tt, N)  1 = was masked
            "memory": memory,                                           # (B, Ts, 1+N, D)  o_1..o_Ts
            "state": state,                                             # (B, 1+N, D)      s_Ts
            "grid": grid,
        }


    def loss(
        self,
        out: dict,
        target: torch.Tensor,
        masked_only: bool = False,
        norm_pix: bool = False,
    ):
        """L2 reconstruction loss. `target`: (B, Tt, 3, H, W).

        Paper default (Sec. 3.1, "Loss"): plain L2 over *all* reconstructed pixels,
        with no patch-level normalization. `masked_only=True` / `norm_pix=True`
        switch to the SiamMAE/MAE convention instead.
        """
        B, Tt = target.shape[:2]
        gt = patchify(target.reshape(B * Tt, *target.shape[2:]), self.patch)
        if norm_pix:
            mu = gt.mean(dim=-1, keepdim=True)
            var = gt.var(dim=-1, keepdim=True)
            gt = (gt - mu) / (var + 1e-6).sqrt()

        pred = out["pred"].reshape(B * Tt, *out["pred"].shape[2:])
        per_patch = (pred.float() - gt.float()).pow(2).mean(dim=-1)

        if not masked_only:
            return per_patch.mean()
        mask = out["mask"].reshape(B * Tt, -1)
        return (per_patch * mask).sum() / mask.sum().clamp(min=1.0)

    def reconstruct_image(self, out: dict) -> torch.Tensor:
        """Predicted patches -> (B, Tt, 3, H, W). Note: only masked patches are supervised."""
        B, Tt, N, _ = out["pred"].shape
        img = unpatchify(out["pred"].reshape(B * Tt, N, -1), self.patch, out["grid"])
        return img.view(B, Tt, *img.shape[1:])


if __name__ == "__main__":
    torch.manual_seed(0)
    model = RVM(encoder_name="facebook/dinov2-small")

    src = torch.randn(2, 4, 3, 224, 224)
    tgt = torch.randn(2, 2, 3, 224, 224)
    deltas = torch.tensor([[4, 8], [2, 6]])

    out = model(src, tgt, deltas)
    print({k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
    print("loss:", model.loss(out, tgt).item())