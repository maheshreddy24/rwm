import dataclasses
import os
import time
from collections.abc import Mapping
from typing import Any, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax.training import train_state
from tqdm import tqdm

from src_jax.models.rvm_jax import RVMConfig, build_model
from src_jax.optimisation.config import TrainerConfig


class TrainState(train_state.TrainState):
    """Adds an EMA copy of the params, updated outside the optimizer/gradient path."""

    ema_params: Any


def _to_numpy_batch(batch):
    """Batch (from `RVMDataset.collate_fn`, NHWC numpy) -> ready for `model.apply`."""
    return {
        "source": batch["source"],
        "target": batch["target"],
        "target_deltas": batch["target_deltas"].astype(np.int32),
    }


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

    Iterates over the *init* tree, so the result always has the full key set the
    current architecture needs. The pretrained checkpoint predates `decoder_proj`,
    so those leaves keep their random init.
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


def build_schedule(config: TrainerConfig, steps_per_epoch: int) -> optax.Schedule:
    """Linear warmup, then cosine decay to `min_lr`."""
    total_steps = max(steps_per_epoch * int(config.num_epochs), 1)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(config.lr),
        warmup_steps=int(config.warmup_steps),
        decay_steps=total_steps,
        end_value=float(config.min_lr),
    )


def build_optimizer(config: TrainerConfig, steps_per_epoch: int):
    schedule = build_schedule(config, steps_per_epoch)
    chain = []
    if config.grad_clip_norm is not None:
        chain.append(optax.clip_by_global_norm(float(config.grad_clip_norm)))
    chain.append(
        optax.adamw(
            learning_rate=schedule,
            b1=float(config.betas[0]),
            b2=float(config.betas[1]),
            eps=float(config.eps),
            weight_decay=float(config.weight_decay),
        )
    )
    return optax.chain(*chain), schedule


def ema_update(ema_params, params, momentum):
    """`ema = momentum * ema + (1 - momentum) * params`, applied leaf-wise."""
    return jax.tree_util.tree_map(
        lambda e, p: momentum * e + (1.0 - momentum) * p, ema_params, params
    )


def representation_loss(out, target_repr, eps=1e-6):
    """Masked MSE between the student's per-patch representation and the EMA target.

    `out['representation']` (B, Tt, N, E) is the decoder's prediction for every patch,
    projected from decoder_emb_dim back to encoder_dim by `decoder_proj`.
    `out['masked_indices']` (B, Tt, N, 1) is 1 for masked/dropped patches, so only
    patches the encoder never saw directly contribute to the loss.

    Targets are instance-normalized over the feature dim (data2vec-style). This is not
    optional: without it the constant solution is trivially reachable and the loss curve
    will look healthy while the encoder collapses.

    Returns:
      (loss, target_std), where target_std is the mean per-dimension std of the
      pre-normalization targets, i.e. the collapse monitor.
    """
    pred = out["representation"].astype(jnp.float32)
    tgt = target_repr.astype(jnp.float32)

    mu = tgt.mean(axis=-1, keepdims=True)
    var = tgt.var(axis=-1, keepdims=True)
    tgt_n = (tgt - mu) * jax.lax.rsqrt(var + eps)

    mask = out["masked_indices"]  # (B, Tt, N, 1)
    per_token = jnp.mean((pred - tgt_n) ** 2, axis=-1, keepdims=True)
    loss = jnp.sum(per_token * mask) / jnp.clip(jnp.sum(mask), min=1.0)

    tgt_std = jnp.mean(jnp.std(tgt.reshape(-1, tgt.shape[-1]), axis=0))
    return loss, tgt_std


def make_train_step(model, ema_momentum):
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
            loss, tgt_std = representation_loss(out, target_repr)
            return loss, tgt_std

        (loss, tgt_std), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        state = state.apply_gradients(grads=grads)
        state = state.replace(
            ema_params=ema_update(state.ema_params, state.params, ema_momentum)
        )

        metrics = {
            "loss": loss,
            "target_std": tgt_std,
            "grad_norm": optax.global_norm(grads),
            "proj_grad_norm": optax.global_norm(grads["decoder_proj"]),
        }
        return state, metrics, next_rng

    return jax.jit(train_step)


def make_eval_step(model):
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
        loss, tgt_std = representation_loss(out, target_repr)
        return {"loss": loss, "target_std": tgt_std}

    return jax.jit(eval_step)


class Trainer:
    def __init__(
        self,
        model_config: RVMConfig,
        train_loader,
        eval_loader=None,
        config: Optional[TrainerConfig] = None,
    ):
        self.model_config = model_config
        self.config = config or TrainerConfig()
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.model = build_model(model_config)

        self.checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"exp_{time.time()}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.ckpt_mgr = ocp.CheckpointManager(
            os.path.abspath(self.checkpoint_dir),
            options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
        )

        self.rng = jax.random.PRNGKey(self.config.seed or 0)
        self.epoch = 0
        self.global_step = 0

        self._init_state()

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
            pretrained = load_pretrained_params(self.config.init_params_path)
            merged = merge_params(params, pretrained)

            ckpt_keys = set(_leaf_paths(pretrained))
            model_keys = set(_leaf_paths(merged))

            orphaned = ckpt_keys - model_keys
            if orphaned:
                raise ValueError(
                    f"checkpoint leaves the model never uses: {sorted(orphaned)}"
                )

            fresh = model_keys - ckpt_keys
            unexpected = {p for p in fresh if p[0] != "decoder_proj"}
            if unexpected:
                raise ValueError(
                    f"unexpected randomly-initialized leaves: {sorted(unexpected)}"
                )

            params = merged
            print(
                f"initialized from {self.config.init_params_path}; "
                f"fresh: {sorted('/'.join(p) for p in fresh)}"
            )

        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"model params: {n_params / 1e6:.1f}M")

        # EMA target encoder starts as an exact copy of the (pretrained) params.
        ema_params = jax.tree_util.tree_map(jnp.array, params)

        steps_per_epoch = len(self.train_loader)
        tx, self.schedule = build_optimizer(self.config, steps_per_epoch)
        self.state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx, ema_params=ema_params
        )

        self.train_step_fn = make_train_step(self.model, float(self.config.ema_momentum))
        self.eval_step_fn = make_eval_step(self.model)

        wandb_config = dataclasses.asdict(self.config) | {
            **dataclasses.asdict(self.model_config),
            "dtype": str(self.model_config.dtype),
        }
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            config=wandb_config,
        )

    def save_checkpoint(self):
        ckpt = {
            "params": self.state.params,
            "ema_params": self.state.ema_params,
            "opt_state": self.state.opt_state,
            "step": self.state.step,
            "epoch": self.epoch,
        }
        self.ckpt_mgr.save(self.global_step, args=ocp.args.StandardSave(ckpt))
        self.ckpt_mgr.wait_until_finished()
        return self.checkpoint_dir

    def resume(self):
        latest = self.ckpt_mgr.latest_step()
        if latest is None:
            return False
        target = {
            "params": self.state.params,
            "ema_params": self.state.ema_params,
            "opt_state": self.state.opt_state,
            "step": self.state.step,
            "epoch": self.epoch,
        }
        restored = self.ckpt_mgr.restore(latest, args=ocp.args.StandardRestore(target))
        self.state = self.state.replace(
            params=restored["params"],
            ema_params=restored["ema_params"],
            opt_state=restored["opt_state"],
            step=restored["step"],
        )
        self.epoch = int(restored["epoch"])
        self.global_step = int(latest)
        return True

    def train(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or int(self.config.num_epochs)
        for _ in tqdm(range(num_epochs), total=num_epochs, leave=False):
            for batch in tqdm(self.train_loader, total=len(self.train_loader), leave=True):
                metrics = self._train_step(batch)
                self.global_step += 1

                if self.global_step % int(self.config.log_interval) == 0:
                    lr = float(self.schedule(self.state.step))
                    print(
                        f"epoch {self.epoch} step {self.global_step} "
                        f"loss {metrics['loss']:.4f} tgt_std {metrics['target_std']:.4f} "
                        f"lr {lr:.2e}"
                    )
                    wandb.log(
                        {f"train/{k}": v for k, v in metrics.items()}
                        | {"train/lr": lr, "epoch": self.epoch},
                        step=self.global_step,
                    )

                if (
                    self.eval_loader is not None
                    and self.global_step % int(self.config.eval_interval) == 0
                ):
                    self._evaluate_and_log()

            self.epoch += 1
            self.save_checkpoint()

            if self.eval_loader is not None:
                self._evaluate_and_log()

    def _train_step(self, batch):
        batch = _to_numpy_batch(batch)
        self.state, metrics, self.rng = self.train_step_fn(self.state, batch, self.rng)
        return {k: float(v) for k, v in metrics.items()}

    def _evaluate_and_log(self):
        metrics = self.eval()
        print(
            f"epoch {self.epoch} step {self.global_step} "
            f"eval_loss {metrics['loss']:.4f} tgt_std {metrics['target_std']:.4f}"
        )
        wandb.log(
            {f"eval/{k}": v for k, v in metrics.items()} | {"epoch": self.epoch},
            step=self.global_step,
        )

    def eval(self):
        totals: dict[str, float] = {}
        n_batches = 0
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