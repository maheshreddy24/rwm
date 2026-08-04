"""EMA / data2vec-style representation trainer (JAX + Flax)."""

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

# checkpoints are named model_{epoch}_{step}.npz; this is the only place that knows it
CKPT_RE = re.compile(r"model_(\d+)_(\d+)\.npz")


class TrainState(train_state.TrainState):
    """Adds an EMA copy of the params, updated outside the optimizer/gradient path."""

    ema_params: Any


_MISSING = object()


# --------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------


def cfg_get(cfg: Any, key: str, default: Any = _MISSING) -> Any:
    """Read `key` from a dataclass, a Mapping, or a plain namespace.

    Only needed for `model_config`, whose concrete type varies; `EMATrainerConfig`
    fields are read directly.
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


def to_numpy_batch(batch):
    """Batch (from `RVMDataset.collate_fn`, NHWC numpy) -> ready for `model.apply`."""
    return {
        "source": batch["source"],
        "target": batch["target"],
        "target_deltas": batch["target_deltas"].astype(np.int32),
    }


def due(step: int, interval) -> bool:
    """Interval check that treats 0/None as 'never' instead of dividing by zero."""
    interval = int(interval or 0)
    return interval > 0 and step % interval == 0


def resolve_dtype(name: str) -> np.dtype:
    """`str(dtype)` -> `np.dtype`, including ml_dtypes extension types.

    `np.dtype("bfloat16")` raises TypeError: numpy's string lookup does not see
    ml_dtypes' registered extension types, and bf16 is the whole reason
    `save_pytree_npz` records per-leaf dtypes at all.
    """
    try:
        return np.dtype(name)
    except TypeError:
        ext = getattr(ml_dtypes, name, None)
        if ext is None:
            raise ValueError(f"unknown dtype {name!r}") from None
        return np.dtype(ext)


# --------------------------------------------------------------------------------------
# checkpoint (de)serialisation
# --------------------------------------------------------------------------------------


def save_pytree_npz(path: str, tree: Any) -> None:
    """Save a pytree to one .npz: leaves as arrays, structure as a pickled treedef.

    Dtype names are stored per-leaf and restored via `.view` on load, because npz
    round-trips ml_dtypes extension types (e.g. bfloat16) as opaque `void` bytes.

    Single-device only: every leaf must be `np.asarray`-convertible on this host.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    # `.view` on load requires a C-contiguous buffer
    arrays = [np.ascontiguousarray(np.asarray(x)) for x in leaves]
    payload = {f"leaf_{i}": arr for i, arr in enumerate(arrays)}
    payload["treedef"] = np.array([pickle.dumps(treedef)], dtype=object)
    payload["dtypes"] = np.array([str(arr.dtype) for arr in arrays], dtype=object)
    # write-then-rename: an interrupted save must not leave a half-written .npz that
    # `resume()` will later try to load
    tmp = f"{path}.tmp.npz"
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def load_pytree_npz(path: str) -> Any:
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


def leaf_paths(p, prefix=()):
    for k, v in flax.core.unfreeze(p).items():
        if isinstance(v, Mapping):
            yield from leaf_paths(v, prefix + (k,))
        else:
            yield prefix + (k,)


def merge_params(init_p, ckpt_p, _prefix=()):
    """Checkpoint values win; freshly-initialized values fill the gaps."""
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


def load_and_merge_params(init_params, path: str, fresh_ok=("decoder_proj",)):
    """Load a flat "/"-joined-key .npz of pretrained params and merge into `init_params`.

    The pretrained checkpoint predates `decoder_proj`, so those leaves keep their random
    init. Any *other* mismatch in either direction means `build_model` has drifted from
    the checkpoint architecture, and is raised rather than silently absorbed.
    """
    flat = np.load(path, allow_pickle=False)
    pretrained: dict = {}
    for key in flat.files:
        parts = key.split("/")
        node = pretrained
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = jnp.asarray(flat[key])

    merged = merge_params(init_params, pretrained)
    ckpt_keys, model_keys = set(leaf_paths(pretrained)), set(leaf_paths(merged))

    orphaned = ckpt_keys - model_keys
    if orphaned:
        raise ValueError(
            f"checkpoint leaves unused by the model, so build_model has drifted from "
            f"the checkpoint architecture: {sorted(orphaned)}"
        )

    fresh = model_keys - ckpt_keys
    unexpected = {p for p in fresh if p[0] not in fresh_ok}
    if unexpected:
        raise ValueError(f"unexpected randomly-initialized leaves: {sorted(unexpected)}")

    print(f"initialized from {path}; fresh: {sorted('/'.join(p) for p in fresh)}")
    return merged


# --------------------------------------------------------------------------------------
# schedules / optimizer
# --------------------------------------------------------------------------------------


def num_optimizer_steps(config: EMATrainerConfig) -> int:
    """`total_steps` counts dataloader batches; the LR/EMA schedules tick once per
    optimizer update, i.e. once every `grad_accum` batches."""
    return max(int(config.total_steps) // int(config.grad_accum), 1)


def build_wd_mask(params, skip_patterns):
    """True (apply weight decay) for every leaf whose path doesn't contain any of
    `skip_patterns` (case-insensitive substring), e.g. 'bias', 'norm', 'cls_token'."""
    patterns = [p.lower() for p in skip_patterns]
    flat = flax.traverse_util.flatten_dict(flax.core.unfreeze(params))
    mask_flat = {
        path: not any(p in "/".join(path).lower() for p in patterns) for path in flat
    }
    # optax's mask must have identical pytree structure, not just equal content
    mask = flax.traverse_util.unflatten_dict(mask_flat)
    return flax.core.freeze(mask) if isinstance(params, flax.core.FrozenDict) else mask


def build_optimizer(config: EMATrainerConfig, params):
    """AdamW + optional clipping, wrapped in MultiSteps, plus the LR schedule.

    Warmup is derived from `total_opt_steps` so it lives on the same axis as
    `decay_steps`; computing it in batch units and dividing by `grad_accum` can round
    to 0 and silently drop warmup entirely.
    """
    total_opt_steps = num_optimizer_steps(config)
    warmup_opt_steps = int(round(total_opt_steps * float(config.warmup_ratio)))
    warmup_opt_steps = max(0, min(warmup_opt_steps, total_opt_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(config.lr_peak),
        warmup_steps=warmup_opt_steps,
        decay_steps=total_opt_steps,
        end_value=float(config.lr_min),
    )

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
            mask=build_wd_mask(params, config.wd_skip),
        )
    )
    tx = optax.MultiSteps(optax.chain(*chain), every_k_schedule=int(config.grad_accum))
    return tx, schedule


def build_ema_momentum_fn(config: EMATrainerConfig):
    """Returns `optimizer_step -> momentum`, where `optimizer_step` is
    `MultiStepsState.gradient_step` (one per real update, not per micro-step)."""
    total_opt_steps = num_optimizer_steps(config)
    start_m, end_m = float(config.ema_momentum), float(config.ema_momentum_end)
    ramp = config.ema_ramp

    def momentum_fn(step):
        if ramp == "cosine":
            t = jnp.clip(step / total_opt_steps, 0.0, 1.0)
            return end_m - (end_m - start_m) * (jnp.cos(jnp.pi * t) + 1.0) / 2.0
        return jnp.asarray(start_m, dtype=jnp.float32)

    return momentum_fn


def ema_update(ema_params, params, momentum, dtype=jnp.float32):
    """`ema = momentum * ema + (1 - momentum) * params`, mixed in fp32, stored in `dtype`.

    The mix must not happen in `dtype`: with bf16 storage and momentum >= 0.999 the
    increment `(1 - m) * (p - e)` sits below one ulp of `e` (bf16 eps ~= 7.8e-3), rounds
    away, and the target encoder silently freezes. fp32 arithmetic removes the
    intermediate rounding; the residual storage-precision risk is warned about in
    `Trainer.__init__`.
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

    Returns (loss, target_std), where target_std is the mean per-dimension std of the
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
    # jnp.maximum, not jnp.clip(..., min=...): the `min=` keyword only exists on newer
    # jax and this is the worst place for a version bump to surface as a TypeError
    loss = jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    tgt_std = jnp.mean(jnp.std(tgt.reshape(-1, tgt.shape[-1]), axis=0))
    return loss, tgt_std


# --------------------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------------------


def reconstruct(model, params, batch, rng):
    """The one masked-reconstruction forward pass, shared by train/eval/collapse."""
    mask_rng, state_rng = jax.random.split(rng)
    return model.apply(
        {"params": params},
        batch["source"],
        batch["target"],
        batch["target_deltas"],
        rng_key=mask_rng,
        rngs={"default": state_rng},
        method=model.reconstruct,
    )


def make_train_step(model, config: EMATrainerConfig, schedule: optax.Schedule):
    momentum_fn = build_ema_momentum_fn(config)
    ema_dtype = getattr(jnp, config.ema_dtype)
    masked_only = bool(config.loss_on_masked_only)

    def train_step(state, batch, rng):
        step_rng, next_rng = jax.random.split(rng)

        def loss_fn(params):
            out = reconstruct(model, params, batch, step_rng)
            target_repr = jax.lax.stop_gradient(
                model.apply(
                    {"params": state.ema_params},
                    batch["target"],
                    method=model.encode_target,
                )
            )
            return representation_loss(out, target_repr, masked_only=masked_only)

        (loss, tgt_std), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # capture the pre-update counter: apply_gradients advances `gradient_step`, so
        # reading it afterwards reports the LR/momentum of the *next* update
        applied_step = state.opt_state.gradient_step
        state = state.apply_gradients(grads=grads)

        # mini_step == 0 means this call just completed a real optimizer update rather
        # than an intermediate grad_accum micro-step
        did_update = state.opt_state.mini_step == 0
        momentum = momentum_fn(applied_step)

        # lax.cond, not momentum=1.0 through a tree_map: on micro-steps that would read,
        # scale and rewrite every EMA leaf just to reproduce its own input
        state = state.replace(
            ema_params=jax.lax.cond(
                did_update,
                lambda: ema_update(state.ema_params, state.params, momentum, ema_dtype),
                lambda: state.ema_params,
            )
        )

        metrics = {
            "loss": loss,
            "target_std": tgt_std,
            "grad_norm": optax.global_norm(grads),
            "ema_momentum": jnp.where(did_update, momentum, 1.0),
            "lr": schedule(applied_step),
            "did_update": did_update.astype(jnp.float32),
        }
        if "decoder_proj" in grads:
            metrics["proj_grad_norm"] = optax.global_norm(grads["decoder_proj"])
        return state, metrics, next_rng

    return jax.jit(train_step)


def make_eval_step(model, masked_only: bool = True):
    def eval_step(params, ema_params, batch, rng):
        out = reconstruct(model, params, batch, rng)
        target_repr = model.apply(
            {"params": ema_params}, batch["target"], method=model.encode_target
        )
        loss, tgt_std = representation_loss(out, target_repr, masked_only=masked_only)
        return {"loss": loss, "target_std": tgt_std}

    return jax.jit(eval_step)


def make_collapse_step(model):
    """Dimensional-collapse diagnostics on the student's predicted representation.

    Separate from `train_step` because the SVD in `collapse_test` is only worth paying
    for every `config.collapse_interval` steps, not every step.
    """

    def collapse_step(params, batch, rng):
        out = reconstruct(model, params, batch, rng)
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
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # epochs is a trainer/schedule concept and lives on EMATrainerConfig
        self.epochs = int(self.config.epochs)
        self.current_epoch = 0
        self.global_step = 0

        self.model = build_model(model_config)
        self.patch_size = tuple(cfg_get(model_config, "patch_size")[-2:])

        self.checkpoint_dir = os.path.join(
            self.config.checkpoint_dir, f"exp_{int(time.time())}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.max_checkpoints_to_keep = 3
        self.rng = jax.random.PRNGKey(self.config.seed or 0)

        # -- keep total_steps (which drives the LR and EMA schedules) equal to the
        # number of batches the loop will actually run, or the cosine decay and
        # momentum ramp finish early / never finish, with nothing to indicate it.
        try:
            self.steps_per_epoch = len(train_loader)
        except TypeError:
            self.steps_per_epoch = None
        configured = int(self.config.total_steps or 0)
        if self.steps_per_epoch is None:
            if configured <= 0:
                raise ValueError(
                    "train_loader has no __len__, so total_steps must be set explicitly"
                )
        else:
            derived = self.steps_per_epoch * self.epochs
            if configured > 0 and configured != derived:
                warnings.warn(
                    f"config.total_steps={configured} but the loop will run "
                    f"{self.epochs} epochs x {self.steps_per_epoch} batches = {derived}; "
                    f"using {derived} so the schedules span the real run.",
                    stacklevel=2,
                )
            if configured != derived:
                if dataclasses.is_dataclass(self.config):
                    self.config = dataclasses.replace(self.config, total_steps=derived)
                else:
                    self.config.total_steps = derived

        # -- warn when ema_dtype cannot represent the EMA increment. The update is
        # e <- e + (1 - m)(p - e); if (1 - m) is below the storage dtype's eps the
        # result rounds back to e and the target encoder stops moving while every
        # logged metric still looks fine.
        if self.config.ema_dtype != "float32":
            try:
                eps = float(np.finfo(resolve_dtype(self.config.ema_dtype)).eps)
            except Exception:
                eps = None
            slowest = 1.0 - max(
                float(self.config.ema_momentum), float(self.config.ema_momentum_end)
            )
            if eps is not None and slowest < eps:
                warnings.warn(
                    f"ema_dtype={self.config.ema_dtype} (eps={eps:.2e}) cannot resolve "
                    f"an EMA increment of {slowest:.2e}; the target encoder may stop "
                    f"updating. Use ema_dtype='float32' unless memory forces otherwise.",
                    stacklevel=2,
                )

        self._init_state()

    def _init_state(self):
        self.rng, init_rng, state_rng = jax.random.split(self.rng, 3)
        first_batch = to_numpy_batch(next(iter(self.train_loader)))

        params = self.model.init(
            {"params": init_rng, "default": state_rng},
            first_batch["source"],
            first_batch["target"],
            first_batch["target_deltas"],
            rng_key=state_rng,
            method=self.model.reconstruct,
        )["params"]

        if self.config.init_params_path is not None:
            params = load_and_merge_params(params, self.config.init_params_path)

        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"model params: {n_params / 1e6:.1f}M")

        # EMA target encoder starts as an exact copy of the (pretrained) params, cast to
        # ema_dtype regardless of the student's compute dtype
        ema_dtype = getattr(jnp, self.config.ema_dtype)
        ema_params = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=ema_dtype), params
        )

        tx, self.schedule = build_optimizer(self.config, params)
        self.state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx, ema_params=ema_params
        )

        masked_only = bool(self.config.loss_on_masked_only)
        self.train_step_fn = make_train_step(self.model, self.config, self.schedule)
        self.eval_step_fn = make_eval_step(self.model, masked_only=masked_only)
        self.collapse_step_fn = make_collapse_step(self.model)

        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            config=cfg_asdict(self.config)
            | {
                **cfg_asdict(self.model_config),
                "dtype": str(cfg_get(self.model_config, "dtype", None)),
            },
        )
        # one x-axis for everything: wandb requires non-decreasing steps, so mixing
        # auto-increment / step=epoch / step=global_step drops points silently
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")

    # -- checkpointing -----------------------------------------------------------------

    def _checkpoints(self) -> list[tuple[int, int, str]]:
        """(epoch, step, path) for every checkpoint in `checkpoint_dir`, oldest first."""
        found = []
        for fname in os.listdir(self.checkpoint_dir):
            m = CKPT_RE.fullmatch(fname)
            if m:
                found.append(
                    (
                        int(m.group(1)),
                        int(m.group(2)),
                        os.path.join(self.checkpoint_dir, fname),
                    )
                )
        return sorted(found)

    def save_checkpoint(self, epoch_done: bool = False) -> str:
        """Write model_{epoch}_{step}.npz and prune all but the newest N."""
        path = os.path.join(
            self.checkpoint_dir, f"model_{self.current_epoch}_{self.global_step}.npz"
        )
        save_pytree_npz(
            path,
            {
                "params": self.state.params,
                "ema_params": self.state.ema_params,
                "opt_state": self.state.opt_state,
                "epoch": np.int32(self.current_epoch),
                "step": np.int32(self.global_step),
                # 1 only for the end-of-epoch save, so resume knows whether to re-run
                # this epoch or move to the next one
                "epoch_done": np.int32(bool(epoch_done)),
                "rng": self.rng,
            },
        )
        if self.max_checkpoints_to_keep > 0:
            for *_, stale in self._checkpoints()[: -self.max_checkpoints_to_keep]:
                os.remove(stale)
        return path

    def resume(self) -> bool:
        ckpts = self._checkpoints()
        if not ckpts:
            return False
        epoch, step, path = ckpts[-1]
        restored = load_pytree_npz(path)
        self.state = self.state.replace(
            params=restored["params"],
            ema_params=restored["ema_params"],
            opt_state=restored["opt_state"],
            step=restored["step"],
        )
        self.global_step = int(step)
        # mid-epoch checkpoints replay their epoch from the start (the loader can't be
        # positioned mid-stream); end-of-epoch ones move on
        self.current_epoch = int(epoch) + int(restored.get("epoch_done", 0))
        if "rng" in restored:
            self.rng = jnp.asarray(restored["rng"], dtype=jnp.uint32)
        print(
            f"resumed from {path} "
            f"(epoch {self.current_epoch}, step {self.global_step})"
        )
        return True

    # -- eval / logging ----------------------------------------------------------------

    def _log(self, payload: dict):
        wandb.log(
            payload | {"epoch": self.current_epoch, "global_step": self.global_step}
        )

    def eval(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        n_batches = 0
        # fixed key on purpose: eval masks should be identical across evaluations so the
        # curve reflects the model, not the draw
        rng = jax.random.PRNGKey(0)
        for batch in tqdm(self.eval_loader, total = len(self.eval_loader)):
            rng, step_rng = jax.random.split(rng)
            metrics = self.eval_step_fn(
                self.state.params,
                self.state.ema_params,
                to_numpy_batch(batch),
                step_rng,
            )
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _eval_and_save(self, epoch_done: bool = False) -> str:
        """Every checkpoint is written at an eval boundary, so model_{epoch}_{step}
        always lines up with a logged eval point."""
        if self.eval_loader is not None:
            metrics = self.eval()
            print(
                f"epoch {self.current_epoch} step {self.global_step} "
                f"eval_loss {metrics['loss']:.4f} tgt_std {metrics['target_std']:.4f}"
            )
            self._log({f"eval/{k}": v for k, v in metrics.items()})
        path = self.save_checkpoint(epoch_done=epoch_done)
        print(f"saved {path}")
        return path

    # -- loop --------------------------------------------------------------------------

    def train(self, max_steps: Optional[int] = None, resume: bool = False):
        """Run the training loop.

        Args:
          max_steps: optional early stop after this many batches. This does *not*
            rescale the LR/EMA schedules — it is a debug cap, not a shorter run.
          resume: pick up from the newest checkpoint in `checkpoint_dir` if one exists.
        """
        if resume:
            self.resume()

        stop = False
        for epoch in tqdm(
            range(self.current_epoch, self.epochs),
            desc="training",
            initial=self.current_epoch,
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
                batch = to_numpy_batch(batch)
                self.state, metrics, self.rng = self.train_step_fn(
                    self.state, batch, self.rng
                )
                # single source of truth: MultiSteps increments state.step once per batch
                self.global_step = int(self.state.step)

                if due(self.global_step, self.config.log_interval):
                    # only materialize here — float() on every step forces a device sync
                    m = {k: float(v) for k, v in metrics.items()}
                    print(
                        f"epoch {epoch} step {self.global_step} "
                        f"loss {m['loss']:.4f} tgt_std {m['target_std']:.4f} "
                        f"lr {m['lr']:.2e} ema_m {m['ema_momentum']:.6f}"
                    )
                    self._log({f"train/{k}": v for k, v in m.items()})

                if due(self.global_step, self.config.collapse_interval):
                    self.rng, step_rng = jax.random.split(self.rng)
                    flat = flatten_collapse_metrics(
                        self.collapse_step_fn(self.state.params, batch, step_rng)
                    )
                    self._log(
                        {f"train/collapse/{k}": float(v) for k, v in flat.items()}
                    )

                if due(self.global_step, self.config.eval_interval):
                    self._eval_and_save()

                if max_steps is not None and self.global_step >= int(max_steps):
                    stop = True
                    break

            batches.close()
            self._eval_and_save(epoch_done=True)

            if stop:
                print(f"stopping early at step {self.global_step} (max_steps)")
                break

        wandb.finish()