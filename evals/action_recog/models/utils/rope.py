"""3-axial rotary position embedding, following DINO-world (Baldassarre et al., 2025, Sec. 3.2).

Every token carries a coordinate triple (tau, i, j): absolute timestamp, and
normalized spatial position on a [-1, +1] grid. These enter attention as a
rotation of q and k rather than an addition to the input, so that
(R(a) q) . (R(b) k) = q . R(b - a) k -- absolute coordinates on both sides
yield relative offsets in the logits.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

RoPE = Tuple[torch.Tensor, torch.Tensor]  # (cos, sin), each (M, N, R)


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


def rope_periods(head_dim: int, t_range, s_range, device) -> torch.Tensor:
    """Angular periods for one head, laid out as (R,) with R = 3 * (head_dim//6).

    Channel pairs are grouped [ tau | i | j ]; any leftover pairs at the end of
    the head are left unrotated, as in DINO-world (4 of 64 dims there).
    """
    per_axis = (head_dim // 2) // 3
    assert per_axis > 0, f"head_dim {head_dim} too small for 3-axial RoPE"

    def logspace(lo, hi):
        return torch.logspace(math.log10(lo), math.log10(hi), per_axis, device=device, dtype=torch.float32)

    return torch.cat([logspace(*t_range), logspace(*s_range), logspace(*s_range)])


def axial_rope(coords: torch.Tensor, periods: torch.Tensor) -> RoPE:
    """coords (M, N, 3) as (tau, i, j)  ->  (cos, sin), each (M, N, R).

    Angle for pair p on axis a is 2*pi * coord_a / period_p, so a "period" is
    the coordinate distance over which that channel pair makes one full turn.
    """
    per_axis = periods.numel() // 3
    ang = coords.repeat_interleave(per_axis, dim=-1)  # (M, N, R): tau..., i..., j...
    ang = 2.0 * math.pi * ang / periods
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, pos: Optional[RoPE]) -> torch.Tensor:
    """Rotate the leading 2R channels of each head. x is (M, H, N, Dh)."""
    if pos is None:
        return x
    cos, sin = pos
    R = cos.shape[-1]
    rot, keep = x[..., : 2 * R], x[..., 2 * R:]
    rot = rot.reshape(*rot.shape[:-1], R, 2)
    c = cos.to(x.dtype).unsqueeze(1).unsqueeze(-1)  # (M, 1, N, R, 1)
    s = sin.to(x.dtype).unsqueeze(1).unsqueeze(-1)
    x0, x1 = rot[..., 0:1], rot[..., 1:2]
    rot = torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)
    return torch.cat([rot.reshape(*x.shape[:-1], 2 * R), keep], dim=-1)
