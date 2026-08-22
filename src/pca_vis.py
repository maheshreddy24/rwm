"""
Per-frame PCA visualization of RVM features.

Input : features of shape (1, T, N, D)  e.g. (1, 6, 256, 512)
        T = frames, N = tokens (16x16 grid), D = channels
Output: one RGB image per frame, where R,G,B = first 3 principal components
        of that frame's token features.
"""

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# core
# ----------------------------------------------------------------------
def pca_3(X, n_components=3):
    """X: (N, D) -> (N, 3) projection onto top principal components."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD is more stable than eig on the covariance matrix
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]                 # (3, D)
    proj = Xc @ comps.T                       # (N, 3)
    evr = (S[:n_components] ** 2) / (S ** 2).sum()
    return proj, comps, evr


def robust_norm(x, lo=2, hi=98):
    """Scale each component to [0,1] using percentiles (kills outliers)."""
    out = np.empty_like(x)
    for c in range(x.shape[-1]):
        a, b = np.percentile(x[..., c], [lo, hi])
        out[..., c] = np.clip((x[..., c] - a) / (b - a + 1e-8), 0, 1)
    return out


def align_signs(proj, ref):
    """PCA sign is arbitrary; flip so components correlate with previous frame."""
    if ref is None:
        return proj
    for c in range(proj.shape[1]):
        if np.dot(proj[:, c], ref[:, c]) < 0:
            proj[:, c] *= -1
    return proj


def pca_per_frame(feats, grid=None, fg_mask=False, keep_sign_consistent=True):
    """
    feats : (1, T, N, D) or (T, N, D), numpy or torch
    grid  : (h, w) token grid; inferred as sqrt(N) if None
    fg_mask: if True, use PC1 to split fore/background and re-run PCA on
             the foreground only (the DINO trick — sharper object colors)
    returns: (T, h, w, 3) float array in [0,1]
    """
    if hasattr(feats, "detach"):                      # torch -> numpy
        feats = feats.detach().float().cpu().numpy()
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 4:
        feats = feats[0]                              # drop batch -> (T, N, D)
    T, N, D = feats.shape

    if grid is None:
        h = int(round(np.sqrt(N)))
        assert h * h == N, f"N={N} is not square, pass grid=(h, w)"
        grid = (h, h)
    h, w = grid

    imgs, ref = [], None
    for t in range(T):
        X = feats[t]                                  # (N, D)
        proj, _, evr = pca_3(X)

        if fg_mask:
            pc1 = proj[:, 0]
            keep = pc1 > np.median(pc1)               # foreground half
            if keep.sum() > 10:
                sub, _, _ = pca_3(X[keep])
                proj = np.zeros((N, 3), np.float32)
                proj[keep] = sub

        if keep_sign_consistent:
            proj = align_signs(proj, ref)
            ref = proj.copy()

        rgb = robust_norm(proj).reshape(h, w, 3)
        imgs.append(rgb)
        print(f"frame {t}: explained variance = "
              f"{np.round(evr, 3)}  (sum {evr.sum():.3f})")

    return np.stack(imgs)                             # (T, h, w, 3)


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def show(imgs, upsample=16, out="rvm_pca.png", frames=None):
    """imgs: (T, h, w, 3). upsample = nearest-neighbour zoom factor."""
    big = np.repeat(np.repeat(imgs, upsample, axis=1), upsample, axis=2)
    T = len(big)
    fig, axes = plt.subplots(1, T, figsize=(3 * T, 3.4))
    axes = np.atleast_1d(axes)
    for t, ax in enumerate(axes):
        ax.imshow(big[t])
        ax.set_title(f"frame {t}")
        ax.axis("off")
    # plt.tight_layout()
    # plt.savefig(out, dpi=140, bbox_inches="tight")
    # print("saved", out)
    return fig


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # replace this with your real tensor: feats = model_output  # (1, 6, 256, 512)
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 256, 512)).astype(np.float32)
    feats = base + 0.35 * rng.normal(size=(1, 6, 256, 512)).astype(np.float32)

    imgs = pca_per_frame(feats)          # (6, 16, 16, 3)
    show(imgs, upsample=16)