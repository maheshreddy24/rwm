#!/usr/bin/env python3
"""Convert an EMA-trainer checkpoint (`save_pytree_npz` layout) into the flat
"/"-joined-key .npz that the downstream eval trainer's `recover_tree` expects.

    python convert_ckpt.py model_12_48000.npz rvm_backbone.npz
    python convert_ckpt.py model_12_48000.npz rvm_backbone.npz --key params
    python convert_ckpt.py model_12_48000.npz --list

Run this from the *training* environment. The checkpoint's treedef is a pickled
`jax.tree_util.PyTreeDef`, and unpickling it reconstructs the optimizer-state
node types (optax `MultiStepsState`, `ScaleByAdamState`, flax `FrozenDict`), so
those packages must be importable and reasonably close to the versions that
wrote the file.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections.abc import Mapping

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    sys.exit("jax is required: the checkpoint stores a pickled jax PyTreeDef.")

try:
    import ml_dtypes  # noqa: F401  -- registers bfloat16 & friends with numpy
except ImportError:
    ml_dtypes = None

# Not referenced directly, but unpickling the treedef needs these classes to
# exist. Import eagerly so a missing package fails here with a clear message
# rather than deep inside pickle.loads.
try:
    import flax  # noqa: F401
    import optax  # noqa: F401
except ImportError as exc:  # pragma: no cover
    print(
        f"warning: {exc.name} not importable; unpickling the treedef will fail "
        f"if the checkpoint contains optimizer state.",
        file=sys.stderr,
    )


# ----------------------------------------------------------------------------
# reading the training-side format
# ----------------------------------------------------------------------------


def resolve_dtype(name: str) -> np.dtype:
    """`str(dtype)` -> `np.dtype`, including ml_dtypes extension types.

    `np.dtype("bfloat16")` raises TypeError: numpy's string lookup does not see
    ml_dtypes' registered extension types, and bf16 is the whole reason
    `save_pytree_npz` records per-leaf dtypes at all.
    """
    try:
        return np.dtype(name)
    except TypeError:
        ext = getattr(ml_dtypes, name, None) if ml_dtypes is not None else None
        if ext is None:
            raise ValueError(
                f"unknown dtype {name!r}"
                + ("" if ml_dtypes is not None else " (ml_dtypes is not installed)")
            ) from None
        return np.dtype(ext)


def load_pytree_npz(path: str):
    """Inverse of `save_pytree_npz`."""
    data = np.load(path, allow_pickle=True)
    treedef = pickle.loads(data["treedef"][0])
    leaves = []
    for i, dtype_str in enumerate(data["dtypes"]):
        arr = data[f"leaf_{i}"]
        target = resolve_dtype(str(dtype_str))
        if arr.dtype != target:
            if arr.dtype.itemsize != target.itemsize:
                raise ValueError(
                    f"leaf_{i}: cannot view {arr.dtype} ({arr.dtype.itemsize}B) "
                    f"as {target} ({target.itemsize}B)"
                )
            arr = arr.view(target)
        leaves.append(jnp.asarray(arr))
    return jax.tree_util.tree_unflatten(treedef, leaves)


# ----------------------------------------------------------------------------
# writing the eval-side format
# ----------------------------------------------------------------------------


def scalar(x) -> int | None:
    """Read a bookkeeping field that may be 0-d or shape (1,).

    `save_pytree_npz` runs `np.ascontiguousarray` over every leaf, and that
    promotes a 0-d array to shape (1,) -- so `epoch`/`step`/`epoch_done` come
    back as length-1 arrays and plain `int()` raises.
    """
    if x is None:
        return None
    flat = np.asarray(jax.device_get(x)).reshape(-1)
    return int(flat[0]) if flat.size else None


def flatten(tree, prefix: str = "") -> dict[str, np.ndarray]:
    """Nested param dict -> `{'a/b/c': np.ndarray}`, the layout `recover_tree` reads."""
    out: dict[str, np.ndarray] = {}
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(v, f"{key}/"))
        else:
            if "/" in str(k):
                raise ValueError(
                    f"param name {k!r} contains '/', which would corrupt the flat "
                    f"key encoding"
                )
            if key in out:
                raise ValueError(f"duplicate flat key {key!r}")
            out[key] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="checkpoint written by save_pytree_npz")
    ap.add_argument("dst", nargs="?", help="output .npz (omit with --list)")
    ap.add_argument(
        "--key",
        default="ema_params",
        help="which subtree to export (default: ema_params; 'params' for the student)",
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        help="cast every leaf to this dtype (default: float32; bf16 does not "
        "survive a plain np.savez round-trip)",
    )
    ap.add_argument("--list", action="store_true", help="print top-level keys and exit")
    args = ap.parse_args()

    ckpt = load_pytree_npz(args.src)

    if not isinstance(ckpt, Mapping):
        sys.exit(f"expected a dict at the top level, got {type(ckpt).__name__}")

    if args.list:
        for k, v in ckpt.items():
            kind = "subtree" if isinstance(v, Mapping) else f"{np.shape(v)} scalar/array"
            print(f"  {k}: {kind}")
        return

    if args.dst is None:
        ap.error("dst is required unless --list is given")

    if args.key not in ckpt:
        sys.exit(
            f"{args.key!r} not in checkpoint; available: {sorted(ckpt)}\n"
            f"(re-run with --list for details)"
        )

    subtree = ckpt[args.key]
    if not isinstance(subtree, Mapping):
        sys.exit(f"{args.key!r} is a leaf, not a param tree")

    target = resolve_dtype(args.dtype)
    flat = {
        k: np.ascontiguousarray(jax.device_get(v)).astype(target, copy=False)
        for k, v in flatten(subtree).items()
    }

    # np.savez with allow_pickle-free readers: every value must be a real numeric
    # array, so fail loudly here rather than at load time in the eval trainer.
    for k, v in flat.items():
        if v.dtype == np.object_:
            sys.exit(f"leaf {k!r} is an object array and cannot be saved flat")

    tmp = f"{args.dst}.tmp.npz"
    np.savez(tmp, **flat)
    os.replace(tmp, args.dst)

    n_params = sum(int(np.prod(v.shape)) for v in flat.values())
    epoch, step = scalar(ckpt.get("epoch")), scalar(ckpt.get("step"))
    where = f" (epoch {epoch}, step {step})" if epoch is not None else ""
    print(
        f"wrote {len(flat)} leaves / {n_params / 1e6:.1f}M params "
        f"from {args.key!r}{where} as {target} -> {args.dst}"
    )

    # sanity: the eval trainer loads with allow_pickle=False, so prove it round-trips
    check = np.load(args.dst, allow_pickle=False)
    assert set(check.files) == set(flat), "round-trip key mismatch"


if __name__ == "__main__":
    main()