"""Attention blocks and the recurrent core built from them."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPE, apply_rope


class MultiHeadAttention(nn.Module):
    """Self- or cross-attention with optional mask and 3-axial RoPE.

    RoPE is applied to q and k only, never to v, and re-applied inside every
    block rather than once at the input -- the rotation has to act on the
    projected queries and keys.
    """

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
        q_pos: Optional[RoPE] = None,
        kv_pos: Optional[RoPE] = None,
    ) -> torch.Tensor:
        if kv is None:
            kv, kv_pos = x, q_pos
        q = apply_rope(self._split(self.q(x)), q_pos)
        k = apply_rope(self._split(self.k(kv)), kv_pos)
        v = self._split(self.v(kv))
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

    def forward(self, x, kv, attn_mask=None, q_pos=None, kv_pos=None):
        x = x + self.ca(self.ca_norm(x), kv, attn_mask, q_pos, kv_pos)
        x = x + self.mlp(self.mlp_norm(x))
        x = x + self.sa(self.sa_norm(x), None, None, q_pos, q_pos)
        return x


class CrossAttentionTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int, mlp_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.head_dim = self.blocks[0].ca.head_dim

    def forward(self, x, kv, attn_mask=None, q_pos=None, kv_pos=None):
        for blk in self.blocks:
            x = blk(x, kv, attn_mask, q_pos, kv_pos)
        return self.out_norm(x)


class GatedRecurrentCore(nn.Module):
    """GRU over a whole token set, cross-attention transformer as candidate.

        z  = sigmoid(W_z x + U_z h)
        r  = sigmoid(W_r x + U_r h)
        h~ = XAttn(q=x, kv=r * LN(h))
        h  <- (1 - z) * h + z * h~

    The candidate's cross-attention is RoPE'd: the incoming frame sits at
    tau_t, the state is tagged tau_{t-1}, so the elapsed gap reaches the core
    through the attention logits. Without this the core sees a gap of 1 and a
    gap of 12 identically.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, mlp_dim: int):
        super().__init__()
        self.input_update = nn.Linear(d_model, d_model, bias=False)
        self.state_update = nn.Linear(d_model, d_model, bias=False)
        self.input_reset = nn.Linear(d_model, d_model, bias=False)
        self.state_reset = nn.Linear(d_model, d_model, bias=False)
        self.state_norm = nn.LayerNorm(d_model, eps=1e-4, bias=False)
        self.transformer = CrossAttentionTransformer(d_model, num_heads, num_layers, mlp_dim)
        self.head_dim = self.transformer.head_dim

    def forward(self, x, state, x_pos=None, state_pos=None):
        z = torch.sigmoid(self.input_update(x) + self.state_update(state))
        r = torch.sigmoid(self.input_reset(x) + self.state_reset(state))
        h = self.transformer(x, r * self.state_norm(state), None, x_pos, state_pos)
        out = (1.0 - z) * state + z * h
        return out, out
