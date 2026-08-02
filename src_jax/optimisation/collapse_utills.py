import numpy as np
import jax
import jax.numpy as jnp


def _effective_rank(x, center=True):
    """Effective rank (RankMe) + explained-variance ranks for a 2D matrix.

    x: (num_samples, e). Samples along axis 0, features (e) along axis 1.
    Returns a dict of scalar diagnostics.
    """
    if center:
        x = x - x.mean(axis=0, keepdims=True)

    # Singular values of the centered matrix; s has length min(num_samples, e).
    # We only need singular values, not U/V -> compute_uv=False is cheaper.
    s = jnp.linalg.svd(x, full_matrices=False, compute_uv=False)

    # RankMe: entropy of the normalized singular-value spectrum, exponentiated.
    # p_i = s_i / sum(s); erank = exp(-sum p_i log p_i).
    eps = 1e-12
    p = s / (s.sum() + eps)
    entropy = -jnp.sum(p * jnp.log(p + eps))
    rankme = jnp.exp(entropy)

    # Explained-variance rank: #components to reach a variance threshold.
    # variance along each singular direction ~ s_i^2
    var = s ** 2
    cum = jnp.cumsum(var) / (var.sum() + eps)
    rank_95 = jnp.sum(cum < 0.95) + 1
    rank_99 = jnp.sum(cum < 0.99) + 1

    return {
        "rankme": rankme,                 # 1..min(n,e); collapse -> toward 1
        "rank_95": rank_95.astype(jnp.float32),
        "rank_99": rank_99.astype(jnp.float32),
        "max_possible": float(min(x.shape)),
    }


def pca_rank_est(vec, max_samples=8192, seed=0):
    """Global dimensional-collapse metric.

    vec: (bs, T, N, e). Flattens (bs, T, N) into samples, keeps e as features.
    Subsamples rows so the SVD stays cheap.
    """
    bs, T, N, e = vec.shape
    flat = vec.reshape(bs * T * N, e)

    # Subsample rows for a cheap SVD.
    n = flat.shape[0]
    if n > max_samples:
        key = jax.random.key(seed)
        idx = jax.random.choice(key, n, shape=(max_samples,), replace=False)
        flat = flat[idx]

    return _effective_rank(flat)


def per_timestep_rank(vec, max_samples=8192, seed=0):
    """RVM-specific: rank at each timestep separately.

    Healthy at t=1 but degrading at later t => recurrence degenerating as it
    unrolls. Invisible in the fully-flattened rank.
    vec: (bs, T, N, e). Returns rankme per timestep, shape (T,).
    """
    bs, T, N, e = vec.shape
    ranks = []
    for t in range(T):
        flat = vec[:, t].reshape(bs * N, e)   # (bs*N, e)
        n = flat.shape[0]
        if n > max_samples:
            key = jax.random.key(seed + t)
            idx = jax.random.choice(key, n, shape=(max_samples,), replace=False)
            flat = flat[idx]
        ranks.append(_effective_rank(flat)["rankme"])
    return jnp.stack(ranks)   # (T,)


def temporal_state_std(vec):
    """RVM-specific: how much does the token change across time?

    Trends to zero => the recurrent state has stopped updating (RNN became a
    pass-through). No analogue in VideoMAE/V-JEPA.
    vec: (bs, T, N, e). Returns a scalar.
    """
    # std across the T axis, then average over everything else.
    return jnp.std(vec, axis=1).mean()


def off_diag_corr(vec):
    """Redundancy metric: mean |off-diagonal| of the e x e feature correlation.

    Rising toward 1 => feature channels becoming copies (informational collapse)
    even if per-dim variance looks fine. This is what Barlow Twins / VICReg
    penalize; here we only measure it.
    vec: (bs, T, N, e) -> correlation over the e feature dims.
    """
    bs, T, N, e = vec.shape
    flat = vec.reshape(bs * T * N, e)          # samples x features

    flat = flat - flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    flat = flat / std

    # correlation matrix over features: (e, e)
    corr = (flat.T @ flat) / flat.shape[0]

    off_diag = corr - jnp.eye(e) * jnp.diag(corr)
    mean_abs_off = jnp.abs(off_diag).sum() / (e * e - e)
    return {
        "mean_abs_offdiag": mean_abs_off,      # 0 = decorrelated, ->1 = collapse
        "max_abs_offdiag": jnp.abs(off_diag).max(),
    }

def collapse_test(vec):
    # vec: bs, t, n, e
    return {
        "global_rank": pca_rank_est(vec),
        "per_t_rank": per_timestep_rank(vec),
        "temporal_std": temporal_state_std(vec),
        "off_diag": off_diag_corr(vec),
    }


def flatten_collapse_metrics(metrics):
    """`collapse_test()`'s nested dict (with scalar/array jax values) -> a flat
    {str: float} dict, ready for `wandb.log`. `per_t_rank` (shape (T,)) expands
    into one `per_t_rank/<t>` entry per timestep."""
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}/{kk}"] = float(vv)
        elif hasattr(v, "shape") and v.shape:
            for i, vv in enumerate(np.asarray(v)):
                flat[f"{k}/{i}"] = float(vv)
        else:
            flat[k] = float(v)
    return flat


if __name__ == "__main__":
    bs, T, N, e = 8, 4, 256, 384
    key = jax.random.key(0)
    random_array = jax.random.normal(key, shape=(bs, T, N, e))

    print("=== healthy (random gaussian) ===")
    print("global rank :", pca_rank_est(random_array))
    print("per-t rank  :", per_timestep_rank(random_array))
    print("temporal std:", float(temporal_state_std(random_array)))
    print("off-diag    :", off_diag_corr(random_array))

    # --- collapse sanity checks: the metrics should react ---
    # (a) dimensional collapse: variance in only a few directions
    low_rank = jax.random.normal(key, (bs, T, N, 4)) @ jax.random.normal(key, (4, e))
    print("\n=== dimensional collapse (rank ~4) ===")
    print("global rank :", pca_rank_est(low_rank))   # rankme should be ~4

    # (b) temporal collapse: state identical across t
    frozen = jnp.broadcast_to(random_array[:, :1], (bs, T, N, e))
    print("\n=== temporal collapse (state frozen across t) ===")
    print("temporal std:", float(temporal_state_std(frozen)))  # -> ~0