"""
EK100 action anticipation probe, following V-JEPA 2.1 (Sec 3.2 / App C.6).

Architecture:
    concat(encoder_tokens, predictor_tokens)  ->  (B, N_enc + N_pred, C)
      -> input projection to probe width D
      -> 3 x self-attention blocks (pre-norm, residual, MLP 4x)
      -> 1 x cross-attention block, 3 learnable queries (action, verb, noun)
         (cross-attn output added back to the query as a residual, then MLP)
      -> 3 independent linear classifiers
    Focal loss (alpha=0.25, gamma=2.0) per classifier, summed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA. Projections are (D -> D); the per-head split happens
    after projection, not before. Heads are moved to dim 1 so the softmax
    runs over the token axis rather than over the head axis."""

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def _split(self, t, B, N):
        # (B, N, D) -> (B, H, N, head_dim)
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        B, N, D = x.shape
        q = self._split(self.query(x), B, N)
        k = self._split(self.key(x), B, N)
        v = self._split(self.value(x), B, N)

        # SDPA rather than an explicit softmax: the context here is ~9k tokens,
        # and materialising a (B, H, N, N) matrix costs several GB per block.
        y = F.scaled_dot_product_attention(q, k, v)     # (B, H, N, head_dim)

        y = y.transpose(1, 2).reshape(B, N, D)          # concat heads
        return self.out(y)


class MultiHeadCrossAttention(nn.Module):
    """Queries and context have independent sequence lengths -- the query
    carries M tokens (3 here) while the context carries N. Nothing about the
    context's length may be inferred from the query's."""

    def __init__(self, dim, kv_dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(kv_dim, dim)
        self.value = nn.Linear(kv_dim, dim)
        self.out = nn.Linear(dim, dim)

    def _split(self, t, B, L):
        return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, context):
        B, M, _ = x.shape
        _, N, _ = context.shape

        q = self._split(self.query(x), B, M)            # (B, H, M, hd)
        k = self._split(self.key(context), B, N)        # (B, H, N, hd)
        v = self._split(self.value(context), B, N)

        y = F.scaled_dot_product_attention(q, k, v)     # (B, H, M, hd), softmax over N

        y = y.transpose(1, 2).reshape(B, M, self.num_heads * self.head_dim)
        return self.out(y)


class Mlp(nn.Sequential):
    def __init__(self, dim, ratio=4):
        super().__init__(
            nn.Linear(dim, dim * ratio),
            nn.GELU(),
            nn.Linear(dim * ratio, dim),
        )


class SelfAttnBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):
    """Final block. The cross-attention output is added back to the query
    token as a residual before the MLP -- this is what makes the queries
    behave as learned pooling slots rather than as a fresh sequence."""

    def __init__(self, dim, kv_dim, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.attn = MultiHeadCrossAttention(dim, kv_dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, q, context):
        q = q + self.attn(self.norm_q(q), self.norm_kv(context))
        q = q + self.mlp(self.norm2(q))
        return q


class ReadOutHead(nn.Module):
    """V-JEPA 2.1 EK100 anticipation probe.

    Args:
        in_dim:        backbone token width (1408 for ViT-g, 1664 for ViT-G)
        dim:           probe width
        num_heads:     attention heads inside the probe
        action_classes: 3568 unique verb-noun pairs in EK100
        verb_classes:   97
        noun_classes:   300
    """

    QUERY_NAMES = ("action", "verb", "noun")

    def __init__(
        self,
        in_dim=1408,
        dim=384,
        num_heads=8,
        depth=3,
        action_classes=3568,
        verb_classes=97,
        noun_classes=300,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim) if in_dim != dim else nn.Identity()

        self.blocks = nn.ModuleList(
            [SelfAttnBlock(dim, num_heads) for _ in range(depth)]
        )

        # three learnable queries: action, verb, noun
        self.queries = nn.Parameter(torch.randn(1, 3, dim) * 0.02)
        self.cross_attn = CrossAttnBlock(dim, kv_dim=dim, num_heads=num_heads)
        self.norm_out = nn.LayerNorm(dim)

        self.action_classifier = nn.Linear(dim, action_classes)
        self.verb_classifier = nn.Linear(dim, verb_classes)
        self.noun_classifier = nn.Linear(dim, noun_classes)

    def forward(self, enc_tokens, pred_tokens=None):
        """
        enc_tokens:  (B, N_enc, C)  frozen encoder output on the context clip
        pred_tokens: (B, N_pred, C) frozen predictor output for the future frame
        """
        if pred_tokens is not None:
            x = torch.cat([enc_tokens, pred_tokens], dim=1)
        else:
            x = enc_tokens

        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)

        q = self.queries.expand(x.shape[0], -1, -1)
        q = self.norm_out(self.cross_attn(q, x))        # (B, 3, D)

        return {
            "action": self.action_classifier(q[:, 0]),
            "verb": self.verb_classifier(q[:, 1]),
            "noun": self.noun_classifier(q[:, 2]),
        }


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    """Multi-class focal loss, alpha=0.25 / gamma=2.0 per App C.6."""
    logp = F.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, target[:, None]).squeeze(1)
    p_t = logp_t.exp()
    return -(alpha * (1.0 - p_t) ** gamma * logp_t).mean()


def anticipation_loss(out, targets):
    """Focal loss applied to each classifier independently, then summed
    before back-propagating through the shared blocks."""
    return sum(focal_loss(out[k], targets[k]) for k in ReadOutHead.QUERY_NAMES)


if __name__ == "__main__":
    B, C = 2, 1408
    N_enc, N_pred = 32 // 2 * 24 * 24, 24 * 24   # tubelet-2 context + 1 future frame

    probe = ReadOutHead(in_dim=C, dim=384, num_heads=8)
    enc = torch.randn(B, N_enc, C)
    pred = torch.randn(B, N_pred, C)

    out = probe(enc, pred)
    for k, v in out.items():
        print(f"{k:>7} logits : {tuple(v.shape)}")

    targets = {
        "action": torch.randint(0, 3568, (B,)),
        "verb": torch.randint(0, 97, (B,)),
        "noun": torch.randint(0, 300, (B,)),
    }
    loss = anticipation_loss(out, targets)
    loss.backward()
    print(f"loss              : {loss.item():.4f}")
    print(f"probe params      : {sum(p.numel() for p in probe.parameters()):,}")
    print("backward pass     : ok")