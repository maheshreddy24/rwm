"""
SSv2 cross-attention readout head from "Scaling 4D Representations" (Carreira et al.).

Structure per supp. sec. A.2 / A.2.1 and Table 6:
    LayerNorm(features) -> + learned temporal embedding -> flatten (T*K)
    -> single learned query cross-attends (768 ch, 12 heads, no out-proj)
    -> residual MLP (hidden = 4x) -> Linear -> 174 logits
"""

import torch
import torch.nn as nn


class CrossAttnHead(nn.Module):
    """One attention head. q comes from the query, k/v from the backbone features.

    No residual here -- the head output is head_dim, not dim, so a residual
    inside the head cannot broadcast. Residual (if any) belongs outside.
    """

    def __init__(self, q_dim, kv_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(q_dim, head_dim)
        self.k = nn.Linear(kv_dim, head_dim)
        self.v = nn.Linear(kv_dim, head_dim)

    def forward(self, q_in, kv_in):
        Q, K, V = self.q(q_in), self.k(kv_in), self.v(kv_in)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5), dim=-1)
        return attn @ V


class MHCrossAttn(nn.Module):
    """Multi-head cross-attention. head_dim * num_heads == q_dim, so the
    concatenation already has width q_dim and no output projection is needed
    (adding one overshoots the Table 6 parameter count)."""

    def __init__(self, q_dim, kv_dim, num_heads):
        super().__init__()
        assert q_dim % num_heads == 0
        head_dim = q_dim // num_heads
        self.heads = nn.ModuleList(
            [CrossAttnHead(q_dim, kv_dim, head_dim) for _ in range(num_heads)]
        )

    def forward(self, q_in, kv_in):
        return torch.cat([h(q_in, kv_in) for h in self.heads], dim=-1)


class Readout(nn.Module):
    """Single-query readout: pools a whole T x K x C clip into one class logit vector.

    Args:
        in_dim:      C, backbone channel width (1024 for ViT-L)
        dim:         D, readout width (768 for SSv2, 1024 for K700)
        num_heads:   12 for SSv2, 16 for K700
        num_frames:  T, 16
        num_classes: 174 for SSv2, 700 for K700
    """

    def __init__(self, in_dim, dim, num_heads, num_frames, num_classes):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        # one embedding per frame, broadcast across the K spatial tokens
        self.temp_emb = nn.Parameter(torch.zeros(1, num_frames, 1, in_dim))
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.xattn = MHCrossAttn(q_dim=dim, kv_dim=in_dim, num_heads=num_heads)
        self.norm_mid = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, feats):
        # feats: (B, T, K, C) from a frozen backbone layer
        B, T, K, C = feats.shape

        x = self.norm_in(feats) + self.temp_emb        # (B, T, K, C)
        x = x.reshape(B, T * K, C)                     # kv sequence

        q = self.query.expand(B, -1, -1)               # (B, 1, D)
        y = self.xattn(q, x)                           # (B, 1, D) -- task features

        y = y + self.mlp(self.norm_mid(y))             # residual MLP
        return self.classifier(y.squeeze(1))           # (B, num_classes)


if __name__ == "__main__":
    # ViT-L backbone: C=1024, 16 frames, 16x16 spatial tokens
    B, T, K, C = 2, 16, 196, 1024

    ssv2 = Readout(in_dim=C, dim=768, num_heads=12, num_frames=T, num_classes=174)
    k700 = Readout(in_dim=C, dim=1024, num_heads=16, num_frames=T, num_classes=700)

    feats = torch.randn(B, T, K, C)
    out = ssv2(feats)

    n_ssv2 = sum(p.numel() for p in ssv2.parameters())
    n_k700 = sum(p.numel() for p in k700.parameters())

    print(f"output shape        : {tuple(out.shape)}   (expected (2, 174))")
    print(f"SSv2 params         : {n_ssv2:,}   (Table 6: 7,041,966)   match={n_ssv2 == 7041966}")
    print(f"K700 params         : {n_k700:,}  (Table 6: 12,281,532)  match={n_k700 == 12281532}")

    out.sum().backward()
    print("backward pass       : ok")