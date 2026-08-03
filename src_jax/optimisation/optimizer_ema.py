"""EMA / data2vec-style representation trainer (JAX + Flax).

Fixes over the previous revision are marked with `# FIX:` comments.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import re
import time
import warnings
from collections.abc import Mapping
from typing import Any, Optional

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import wandb
from flax.training import train_state
from tqdm import tqdm

from src_jax.models.rvm_jax import RVMConfig, build_model
from src_jax.optimisation.collapse_utills import collapse_test, flatten_collapse_metrics
from src_jax.optimisation.config import EMATrainerConfig


class TrainState(train_state.TrainState):
    """Adds an EMA copy of the params, updated outside the optimizer/gradient path."""

    ema_params: Any


# --------------------------------------------------------------------------------------
# config access
# --------------------------------------------------------------------------------------

_MISSING = object()


def cfg_get(cfg: Any, key: str, default: Any = _MISSING) -> Any:
    """Read `key` from a dataclass, a Mapping, or a plain namespace.

    FIX: the previous revision mixed `model_config['epochs']` with
    `model_config.patch_size`, so exactly one of the two was guaranteed to fail
    depending on what `RVMConfig` actually is. Going through one accessor makes the
    trainer agnostic to that choice.
    """
    if isinstance(cfg, Mapping):
        if key in cfg:
            return cfg[key]
    elif hasattr(cfg, key):
        return getattr(cfg, key)
    if default is _MISSING:
        raise KeyError(f"{type(cfg).__name__} has no field {key!r}")
    return default


def cfg_asdict(cfg: Any) -> dict:
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}


def _to_numpy_batch(batch):
    """Batch (from `RVMDataset.collate_fn`, NHWC numpy) -> ready for `model.apply`."""
    return {
        "source": batch["source"],
        "target": batch["target"],
        "target_deltas": batch["target_deltas"].astype(np.int32),
    }


# --------------------------------------------------------------------------------------
# checkpoint (de)serialisation
# --------------------------------------------------------------------------------------


def resolve_dtype(name: str) -> np.dtype:
    """`str(dtype)` -> `np.dtype`, including ml_dtypes extension types.

    FIX: `np.dtype("bfloat16")` raises `TypeError` — numpy's string lookup does not
    see ml_dtypes' registered extension types. Since bf16 is the whole reason
    `save_pytree_npz` records per-leaf dtypes, the reload path was broken for exactly
    the case it existed to serve.
    """
    try:
        return np.dtype(name)
    except TypeError:
        ext = getattr(ml_dtypes, name, None)
        if ext is None:
            raise ValueError(f"unknown dtype {name!r}") from None
        return np.dtype(ext)


def load_pretrained_params(path: str) -> dict:
    """Load a Flax params pytree saved as a flat "/"-joined-key .npz."""
    flat = np.load(path, allow_pickle=False)
    tree: dict = {}
    for key in flat.files:
        parts = key.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = jnp.asarray(flat[key])
    return tree


def merge_params(init_p, ckpt_p, _prefix=()):
    """Checkpoint values win; freshly-initialized values fill the gaps.

    The pretrained checkpoint predates `decoder_proj`, so those leaves keep their
    random init. Any other fresh leaf means `build_model` has drifted from the
    checkpoint's architecture, which the caller surfaces rather than swallows.
    """
    out = {}
    for k, v in flax.core.unfreeze(init_p).items():
        path = _prefix + (k,)
        if k not in ckpt_p:
            out[k] = v
        elif isinstance(v, Mapping):
            out[k] = merge_params(v, ckpt_p[k], path)
        else:
            c = jnp.asarray(ckpt_p[k])
            if c.shape != jnp.shape(v):
                raise ValueError(
                    f"{'/'.join(path)}: checkpoint {c.shape} vs model {jnp.shape(v)}"
                )
            out[k] = c
    return out


def _leaf_paths(p, prefix=()):
    for k, v in flax.core.unfreeze(p).items():
        if isinstance(v, Mapping):
            yield from _leaf_paths(v, prefix + (k,))
        else:
            yield prefix + (k,)


def save_pytree_npz(path: str, tree: Any) -> None:
    """Save an arbitrary pytree (nested dicts/namedtuples of arrays/scalars) to a
    single .npz file: leaves as arrays, structure as a pickled treedef alongside them.

    Dtype names are stored per-leaf and restored via `.view` on load: npz round-trips
    ml_dtypes extension types (e.g. bfloat16, used for `ema_dtype`) as opaque `void`
    bytes rather than preserving the dtype, so a plain `jnp.asarray` on load would
    silently corrupt them.

    Single-device only: every leaf must already live on this host as something
    `np.asarray` can convert directly (no cross-device gather/reshard).
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    # FIX: `.view` on load requires a C-contiguous buffer; `np.ascontiguousarray`
    # makes that explicit rather than relying on how the leaf happened to be laid out.
    arrays = [np.ascontiguousarray(np.asarray(x)) for x in leaves]
    payload = {f"leaf_{i}": arr for i, arr in enumerate(arrays)}
    payload["treedef"] = np.array([pickle.dumps(treedef)], dtype=object)
    payload["dtypes"] = np.array([str(arr.dtype) for arr in arrays], dtype=object)
    # FIX: write-then-rename, so an interrupted save can't leave a half-written .npz
    # that `_checkpoint_steps` will later happily try to resume from. `np.savez`
    # appends '.npz' unless the name already ends in it, hence the explicit suffix.
    tmp = f"{path}.tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def load_pytree_npz(path: str) -> Any:
    """Inverse of `save_pytree_npz`."""
    data = np.load(path, allow_pickle=True)
    treedef = pickle.loads(data["treedef"][0])
    dtypes = data["dtypes"]
    leaves = []
    for i, dtype_str in enumerate(dtypes):
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


# --------------------------------------------------------------------------------------
# schedules / optimizer
# --------------------------------------------------------------------------------------


def num_optimizer_steps(config: EMATrainerConfig) -> int:
    """`total_steps` counts dataloader batches; the LR/EMA schedules tick once per
    optimizer update, i.e. once every `grad_accum` batches."""
    return max(int(config.total_steps) // int(config.grad_accum), 1)


def build_schedule(config: EMATrainerConfig) -> optax.Schedule:
    """Linear warmup, then cosine decay to `lr_min`, indexed by optimizer step."""
    total_opt_steps = num_optimizer_steps(config)
    # FIX: warmup is now derived from `total_opt_steps` so it lives on the same axis as
    # `decay_steps`; the old `round(total_steps * ratio) // grad_accum` could round to 0
    # for small ratios and silently drop warmup entirely.
    warmup_opt_steps = int(round(total_opt_steps * float(config.warmup_ratio)))
    warmup_opt_steps = max(0, min(warmup_opt_steps, total_opt_steps))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(config.lr_peak),
        warmup_steps=warmup_opt_steps,
        decay_steps=total_opt_steps,
        end_value=float(config.lr_min),
    )


def build_wd_mask(params, skip_patterns):
    """True (apply weight decay) for every leaf whose path doesn't contain any of
    `skip_patterns` (case-insensitive substring match), e.g. 'bias', 'norm',
    'cls_token', 'pos_embed' stay undecayed."""
    patterns = [p.lower() for p in skip_patterns]
    flat = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
    mask_flat = {
        path: not any(p in "/".join(path).lower() for p in patterns) for path in flat
    }
    # Match `params`' own container type (plain dict on current flax versions;
    # optax's mask must have identical pytree structure, not just equal content).
    mask = flax.traverse_util.unflatten_dict(mask_flat)
    return flax.core.freeze(mask) if isinstance(params, flax.core.FrozenDict) else mask


def build_optimizer(config: EMATrainerConfig, params):
    schedule = build_schedule(config)
    mask = build_wd_mask(params, config.wd_skip)
    chain = []
    if config.grad_clip is not None:
        chain.append(optax.clip_by_global_norm(float(config.grad_clip)))
    chain.append(
        optax.adamw(
            learning_rate=schedule,
            b1=float(config.betas[0]),
            b2=float(config.betas[1]),
            eps=float(config.eps),
            weight_decay=float(config.weight_decay),
            mask=mask,
        )
    )
    inner = optax.chain(*chain)
    tx = optax.MultiSteps(inner, every_k_schedule=int(config.grad_accum))
    return tx, schedule


def build_ema_momentum_fn(config: EMATrainerConfig):
    """Returns `optimizer_step -> momentum`. `optimizer_step` is `MultiStepsState.gradient_step`
    (increments once per real optimizer update, not per accumulation micro-step)."""
    total_opt_steps = num_optimizer_steps(config)
    start_m = float(config.ema_momentum)
    end_m = float(config.ema_momentum_end)
    ramp = config.ema_ramp

    def momentum_fn(step):
        if ramp == "cosine":
            t = jnp.clip(step / total_opt_steps, 0.0, 1.0)
            return end_m - (end_m - start_m) * (jnp.cos(jnp.pi * t) + 1.0) / 2.0
        return jnp.asarray(start_m, dtype=jnp.float32)

    return momentum_fn


def ema_update(ema_params, params, momentum, dtype=jnp.float32):
    """`ema = momentum * ema + (1 - momentum) * params`, accumulated in float32 and
    stored in `dtype`.

    FIX: the arithmetic was previously done in `dtype`. With bf16 storage and
    momentum >= 0.999 the increment `(1 - m) * (p - e)` sits below one ulp of `e`
    (bf16 eps ~= 7.8e-3), so it rounds away and the target encoder silently freezes.
    Doing the mix in fp32 removes the intermediate rounding; see the check in
    `Trainer._check_ema_precision` for the residual storage-precision risk.
    """
    m = jnp.asarray(momentum, dtype=jnp.float32)
    return jax.tree_util.tree_map(
        lambda e, p: (
            m * e.astype(jnp.float32) + (1.0 - m) * p.astype(jnp.float32)
        ).astype(dtype),
        ema_params,
        params,
    )


# --------------------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------------------


def representation_loss(out, target_repr, masked_only=True, eps=1e-6):
    """Masked MSE between the student's per-patch representation and the EMA target.

    `out['representation']` (B, Tt, N, E) is the decoder's prediction for every patch,
    projected from decoder_emb_dim back to encoder_dim by `decoder_proj`.
    `out['masked_indices']` (B, Tt, N, 1) is 1 for masked/dropped patches, so only
    patches the encoder never saw directly contribute to the loss.

    Targets are instance-normalized over the feature dim (data2vec-style). This is not
    optional: without it the constant solution is trivially reachable and the loss curve
    will look healthy while the encoder collapses.

    Returns:
      (loss, target_std) where target_std is the mean per-dimension std of the
      pre-normalization targets, i.e. the collapse monitor.
    """
    pred = out["representation"].astype(jnp.float32)  # B, T, N, E
    tgt = target_repr.astype(jnp.float32)  # B, T, N, E

    mu = tgt.mean(axis=-1, keepdims=True)
    var = tgt.var(axis=-1, keepdims=True)
    tgt_n = (tgt - mu) * jax.lax.rsqrt(var + eps)

    mask = out["masked_indices"] if masked_only else jnp.ones_like(out["masked_indices"])
    mask = mask.astype(jnp.float32)
    per_token = jnp.mean((pred - tgt_n) ** 2, axis=-1, keepdims=True)
    # FIX: `jnp.maximum` instead of `jnp.clip(..., min=...)` — the `min=` keyword is
    # only available on newer jax, and this is the one place a version bump would turn
    # into a confusing TypeError deep inside a jit trace.
    loss = jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    tgt_std = jnp.mean(jnp.std(tgt.reshape(-1, tgt.shape[-1]), axis=0))
    return loss, tgt_std


# --------------------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------------------


def make_train_step(model, config: EMATrainerConfig, schedule: optax.Schedule):
    momentum_fn = build_ema_momentum_fn(config)
    ema_dtype = getattr(jnp, config.ema_dtype)
    masked_only = bool(config.loss_on_masked_only)

    def train_step(state, batch, rng):
        mask_rng, state_rng, next_rng = jax.random.split(rng, 3)

        def loss_fn(params):
            out = model.apply(
                {"params": params},
                batch["source"],
                batch["target"],
                batch["target_deltas"],
                rng_key=mask_rng,
                rngs={"default": state_rng},
                method=model.reconstruct,
            )
            target_repr = jax.lax.stop_gradient(
                model.apply(
                    {"params": state.ema_params},
                    batch["target"],
                    method=model.encode_target,
                )
            )
            loss, tgt_std = representation_loss(out, target_repr, masked_only=masked_only)
            return loss, tgt_std

        (loss, tgt_std), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # FIX: capture the pre-update counters. `apply_gradients` advances
        # `gradient_step`, so reading it afterwards reports the LR/momentum of the
        # *next* update rather than the one that was just applied.
        prev_opt_state = state.opt_state
        applied_step = prev_opt_state.gradient_step

        state = state.apply_gradients(grads=grads)

        # `mini_step == 0` means this call just completed a real optimizer update
        # (as opposed to an intermediate grad_accum accumulation step); gate the
        # EMA update on it, and ramp momentum over `gradient_step` (the real-update
        # counter), not the per-batch `state.step`.
        did_update = state.opt_state.mini_step == 0
        momentum = momentum_fn(applied_step)

        # FIX: `lax.cond` instead of folding momentum=1.0 through a full tree_map —
        # on accumulation micro-steps the old code still read, scaled and rewrote
        # every EMA leaf just to reproduce its own input.
        new_ema = jax.lax.cond(
            did_update,
            lambda: ema_update(state.ema_params, state.params, momentum, ema_dtype),
            lambda: state.ema_params,
        )
        state = state.replace(ema_params=new_ema)

        metrics = {
            "loss": loss,
            "target_std": tgt_std,
            "grad_norm": optax.global_norm(grads),
            "ema_momentum": jnp.where(did_update, momentum, 1.0),
            "lr": schedule(applied_step),
            "did_update": did_update.astype(jnp.float32),
        }
        # FIX: guarded — `grads["decoder_proj"]` was an unconditional KeyError for any
        # model built without that head.
        if "decoder_proj" in grads:
            metrics["proj_grad_norm"] = optax.global_norm(grads["decoder_proj"])
        return state, metrics, next_rng

    return jax.jit(train_step)


def make_eval_step(model, masked_only: bool = True):
    def eval_step(params, ema_params, batch, rng):
        mask_rng, state_rng = jax.random.split(rng)
        out = model.apply(
            {"params": params},
            batch["source"],
            batch["target"],
            batch["target_deltas"],
            rng_key=mask_rng,
            rngs={"default": state_rng},
            method=model.reconstruct,
        )
        target_repr = model.apply(
            {"params": ema_params}, batch["target"], method=model.encode_target
        )
        loss, tgt_std = representation_loss(out, target_repr, masked_only=masked_only)
        return {"loss": loss, "target_std": tgt_std}

    return jax.jit(eval_step)


def make_collapse_step(model):
    """Dimensional-collapse diagnostics on the student's predicted representation.

    Kept separate from `train_step` (rather than folded into `representation_loss`)
    because the SVD in `collapse_test` is only worth paying for every
    `config.collapse_interval` steps, not every step.
    """

    def collapse_step(params, batch, rng):
        mask_rng, state_rng = jax.random.split(rng)
        out = model.apply(
            {"params": params},
            batch["source"],
            batch["target"],
            batch["target_deltas"],
            rng_key=mask_rng,
            rngs={"default": state_rng},
            method=model.reconstruct,
        )
        return collapse_test(out["representation"].astype(jnp.float32))

    return jax.jit(collapse_step)


# --------------------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------------------


class Trainer:
    def __init__(
        self,
        model_config: RVMConfig,
        train_loader,
        eval_loader=None,
        config: Optional[EMATrainerConfig] = None,
    ):
        self.model_config = model_config
        self.config = config or EMATrainerConfig()
        # FIX: was `cfg_get(model_config, "epochs")` -- epochs is a trainer/schedule
        # concept and lives on `config` (EMATrainerConfig), not the model config.
        self.epochs = int(cfg_get(self.config, "epochs"))
        self.current_epoch = 0
        # FIX: `global_step` was only ever assigned inside `resume()`, so a fresh run
        # raised AttributeError the first time it checkpointed or evaluated.
        self.global_step = 0
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.model = build_model(model_config)
        self.patch_size = tuple(cfg_get(model_config, "patch_size")[-2:])

        self.checkpoint_dir = os.path.join(
            self.config.checkpoint_dir, f"exp_{int(time.time())}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.max_checkpoints_to_keep = 3

        self.rng = jax.random.PRNGKey(self.config.seed or 0)

        self._reconcile_total_steps()
        self._init_state()

    # -- setup -------------------------------------------------------------------------

    def _reconcile_total_steps(self):
        """Keep `config.total_steps` (which drives the LR and EMA schedules) in sync
        with the number of batches the loop will actually run.

        FIX: previously the loop ran `epochs * len(train_loader)` batches while the
        schedules were built from an independently-configured `total_steps`. Any
        mismatch meant the cosine decay and momentum ramp finished early or never
        finished at all, with nothing to indicate it.
        """
        try:
            per_epoch = len(self.train_loader)
        except TypeError:
            per_epoch = None

        configured = int(cfg_get(self.config, "total_steps", 0) or 0)

        if per_epoch is None:
            if configured <= 0:
                raise ValueError(
                    "train_loader has no __len__, so total_steps must be set explicitly"
                )
            self.steps_per_epoch = None
            return

        self.steps_per_epoch = per_epoch
        derived = per_epoch * self.epochs
        if configured > 0 and configured != derived:
            warnings.warn(
                f"config.total_steps={configured} but the loop will run "
                f"{self.epochs} epochs x {per_epoch} batches = {derived}; "
                f"using {derived} so the schedules span the real run.",
                stacklevel=2,
            )
        if configured != derived:
            self.config = self._with_total_steps(derived)

    def _with_total_steps(self, total_steps: int):
        if dataclasses.is_dataclass(self.config):
            return dataclasses.replace(self.config, total_steps=total_steps)
        self.config.total_steps = total_steps
        return self.config

    def _check_ema_precision(self):
        """Warn when `ema_dtype` cannot represent the EMA increment.

        The update is `e <- e + (1 - m)(p - e)`. If `(1 - m)` is below the storage
        dtype's eps, the result rounds back to `e` and the target encoder stops moving
        while every logged metric still looks fine.
        """
        if self.config.ema_dtype == "float32":
            return
        try:
            eps = float(np.finfo(resolve_dtype(self.config.ema_dtype)).eps)
        except Exception:
            return
        slowest = 1.0 - max(
            float(self.config.ema_momentum), float(self.config.ema_momentum_end)
        )
        if slowest < eps:
            warnings.warn(
                f"ema_dtype={self.config.ema_dtype} (eps={eps:.2e}) cannot resolve an "
                f"EMA increment of {slowest:.2e}; the target encoder may stop updating. "
                f"Use ema_dtype='float32' unless memory forces otherwise.",
                stacklevel=2,
            )

    def _init_state(self):
        self.rng, init_rng, state_rng = jax.random.split(self.rng, 3)
        first_batch = _to_numpy_batch(next(iter(self.train_loader)))

        params = self.model.init(
            {"params": init_rng, "default": state_rng},
            first_batch["source"],
            first_batch["target"],
            first_batch["target_deltas"],
            rng_key=state_rng,
            method=self.model.reconstruct,
        )["params"]

        if self.config.init_params_path is not None:
            params = self._load_and_merge(params)

        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"model params: {n_params / 1e6:.1f}M")

        self._check_ema_precision()

        # EMA target encoder starts as an exact copy of the (pretrained) params, cast
        # to `ema_dtype` regardless of the student's compute dtype.
        ema_dtype = getattr(jnp, self.config.ema_dtype)
        ema_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=ema_dtype), params)

        tx, self.schedule = build_optimizer(self.config, params)
        self.state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx, ema_params=ema_params
        )

        self.train_step_fn = make_train_step(self.model, self.config, self.schedule)
        self.eval_step_fn = make_eval_step(
            self.model, masked_only=bool(self.config.loss_on_masked_only)
        )
        self.collapse_step_fn = make_collapse_step(self.model)

        wandb_config = cfg_asdict(self.config) | {
            **cfg_asdict(self.model_config),
            "dtype": str(cfg_get(self.model_config, "dtype", None)),
        }
        wandb.init(
            project=self.config.wandb_project, name=self.config.wandb_name, config=wandb_config
        )
        # FIX: all logging now shares one x-axis. The old code mixed auto-increment
        # (train), `step=epoch` (collapse) and `step=global_step` (eval); wandb requires
        # non-decreasing steps, so the collapse points were being dropped silently.
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")

    def _load_and_merge(self, init_params):
        """Fill from the pretrained checkpoint; `decoder_proj` stays randomly initialized."""
        pretrained = load_pretrained_params(self.config.init_params_path)
        merged = merge_params(init_params, pretrained)

        ckpt_keys = set(_leaf_paths(pretrained))
        model_keys = set(_leaf_paths(merged))

        orphaned = ckpt_keys - model_keys
        if orphaned:
            raise ValueError(
                f"checkpoint leaves unused by the model, so build_model has drifted "
                f"from the checkpoint architecture: {sorted(orphaned)}"
            )

        fresh = model_keys - ckpt_keys
        expected_fresh = {p for p in fresh if p[0] == "decoder_proj"}
        if fresh != expected_fresh:
            raise ValueError(
                f"unexpected randomly-initialized leaves: {sorted(fresh - expected_fresh)}"
            )

        print(
            f"initialized from {self.config.init_params_path}; "
            f"fresh: {sorted('/'.join(p) for p in fresh)}"
        )
        return merged

    # -- checkpointing -----------------------------------------------------------------

    def _checkpoint_path(self, step: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ckpt_{step:08d}.npz")

    def _checkpoint_steps(self) -> list[int]:
        steps = []
        for fname in os.listdir(self.checkpoint_dir):
            m = re.fullmatch(r"ckpt_(\d+)\.npz", fname)
            if m:
                steps.append(int(m.group(1)))
        return sorted(steps)

    def save_checkpoint(self, epoch: int) -> str:
        ckpt = {
            "params": self.state.params,
            "ema_params": self.state.ema_params,
            "opt_state": self.state.opt_state,
            "step": self.state.step,
            # FIX: store the *completed* epoch and resume at epoch + 1; the old field
            # was ambiguous and `train()` restarted from epoch 0 regardless.
            "epoch": epoch,
            "global_step": self.global_step,
            # FIX: the sampling rng was not checkpointed, so a resumed run replayed
            # the same mask/dropout draws from the very beginning.
            "rng": self.rng,
        }
        path = self._checkpoint_path(self.global_step)
        save_pytree_npz(path, ckpt)

        for stale in self._checkpoint_steps()[: -self.max_checkpoints_to_keep]:
            os.remove(self._checkpoint_path(stale))
        return path

    def resume(self) -> bool:
        steps = self._checkpoint_steps()
        if not steps:
            return False
        latest = steps[-1]
        restored = load_pytree_npz(self._checkpoint_path(latest))
        self.state = self.state.replace(
            params=restored["params"],
            ema_params=restored["ema_params"],
            opt_state=restored["opt_state"],
            step=restored["step"],
        )
        self.current_epoch = int(restored["epoch"]) + 1
        self.global_step = int(restored.get("global_step", latest))
        if "rng" in restored:
            self.rng = jnp.asarray(restored["rng"], dtype=jnp.uint32)
        print(
            f"resumed from {self._checkpoint_path(latest)} "
            f"(epoch {self.current_epoch}, step {self.global_step})"
        )
        return True

    # -- loop --------------------------------------------------------------------------

    @staticmethod
    def _due(step: int, interval) -> bool:
        """Interval check that tolerates 0/None as 'never' instead of dividing by zero."""
        interval = int(interval or 0)
        return interval > 0 and step % interval == 0

    def train(self, max_steps: Optional[int] = None, resume: bool = False):
        """Run the training loop.

        Args:
          max_steps: optional early stop after this many batches. Note this does *not*
            rescale the LR/EMA schedules — it is a debug cap, not a shorter run.
          resume: pick up from the newest checkpoint in `checkpoint_dir` if one exists.
        """
        if resume:
            self.resume()

        stop = False
        for epoch in tqdm(
            range(self.current_epoch, self.epochs), desc="training", initial=self.current_epoch,
            total=self.epochs,
        ):
            self.current_epoch = epoch
            batches = tqdm(
                self.train_loader,
                desc=f"epoch {epoch}",
                total=self.steps_per_epoch,
                leave=False,
            )
            for batch in batches:
                metrics = self._train_step(batch)

                # FIX: intervals are keyed off the global step, not the in-epoch step,
                # so they no longer all fire at step 0 of every epoch (which meant an
                # eval pass before any training had happened).
                if self._due(self.global_step, self.config.log_interval):
                    print(
                        f"epoch {epoch} step {self.global_step} "
                        f"loss {metrics['loss']:.4f} tgt_std {metrics['target_std']:.4f} "
                        f"lr {metrics['lr']:.2e} ema_m {metrics['ema_momentum']:.6f}"
                    )
                    self._log({f"train/{k}": v for k, v in metrics.items()}, epoch)

                if self._due(self.global_step, self.config.collapse_interval):
                    self._log_collapse_metrics(batch, epoch)

                if self.eval_loader is not None and self._due(
                    self.global_step, self.config.eval_interval
                ):
                    self._evaluate_and_log(epoch)

                if max_steps is not None and self.global_step >= int(max_steps):
                    stop = True
                    break

            batches.close()
            self.save_checkpoint(epoch)

            if self.eval_loader is not None:
                self._evaluate_and_log(epoch)

            if stop:
                print(f"stopping early at step {self.global_step} (max_steps)")
                break

        wandb.finish()

    def _log(self, payload: dict, epoch: int):
        wandb.log(payload | {"epoch": epoch, "global_step": self.global_step})

    def _train_step(self, batch):
        batch = _to_numpy_batch(batch)
        self.state, metrics, self.rng = self.train_step_fn(self.state, batch, self.rng)
        # FIX: `global_step` is now advanced from the single source of truth
        # (`state.step`, which MultiSteps increments once per batch) rather than never
        # being updated at all — checkpoints previously all wrote to ckpt_00000000.npz.
        self.global_step = int(self.state.step)
        return {k: float(v) for k, v in metrics.items()}

    def _log_collapse_metrics(self, batch, epoch: int):
        batch = _to_numpy_batch(batch)
        self.rng, step_rng = jax.random.split(self.rng)
        metrics = self.collapse_step_fn(self.state.params, batch, step_rng)
        flat = flatten_collapse_metrics(metrics)
        self._log({f"train/collapse/{k}": float(v) for k, v in flat.items()}, epoch)

    def _evaluate_and_log(self, epoch: int):
        metrics = self.eval()
        print(
            f"epoch {epoch} step {self.global_step} "
            f"eval_loss {metrics['loss']:.4f} tgt_std {metrics['target_std']:.4f}"
        )
        self._log({f"eval/{k}": v for k, v in metrics.items()}, epoch)

    def eval(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        n_batches = 0
        # Fixed key on purpose: eval masks should be identical across evaluations so the
        # curve reflects the model, not the draw.
        rng = jax.random.PRNGKey(0)
        for batch in self.eval_loader:
            batch = _to_numpy_batch(batch)
            rng, step_rng = jax.random.split(rng)
            metrics = self.eval_step_fn(
                self.state.params, self.state.ema_params, batch, step_rng
            )
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in totals.items()}