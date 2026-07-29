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
from PIL import Image
from tqdm import tqdm

from src_jax.models.rvm_jax import RVMConfig, build_model, patchify, rvm_loss, unpatchify
from src_jax.optimisation.config import TrainerConfig


class TrainState(train_state.TrainState):
    """Adds an EMA copy of the params, updated outside the optimizer/gradient path."""

    ema_params: Any


# --------------------------------------------------------------------------------------
# batch / pytree helpers
# --------------------------------------------------------------------------------------


def _to_numpy_batch(batch):
    """Torch batch (from `RVMDataset.collate_fn`, NHWC) -> numpy, ready for `model.apply`."""
    return {
        "source": batch["source"].numpy(),
        "target": batch["target"].numpy(),
        "target_deltas": batch["target_deltas"].numpy().astype(np.int32),
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

    The pretrained checkpoint predates `decoder_proj`, so those leaves keep their
    random init. Any *other* fresh leaf means `build_model` has drifted from the
    checkpoint's architecture -- surfaced by the caller rather than swallowed here.
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


# --------------------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------------------


def build_schedule(config: TrainerConfig, steps_per_epoch: int) -> optax.Schedule:
    """Linear warmup -> cosine decay to `min_lr`."""
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


# --------------------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------------------


def representation_loss(out, target_repr, eps=1e-6):
    """Masked MSE between the student's per-patch representation and the EMA target.

    `out['representation']` (B, Tt, N, E) is the decoder's prediction for every patch,
    projected from decoder_emb_dim back to encoder_dim by `decoder_proj`.
    `out['masked_indices']` (B, Tt, N, 1) is 1 for masked/dropped patches, so -- as in
    MAE/SiamMAE's pixel loss -- only patches the encoder never saw contribute.

    Targets are instance-normalized over the feature dim (data2vec-style). This is not
    optional: without it the constant solution is trivially reachable and the loss curve
    will look healthy while the encoder collapses.

    Returns:
      (loss, target_std) where target_std is the mean per-dimension std of the
      *pre-normalization* targets -- the collapse monitor.
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


# --------------------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------------------


def make_train_step(model, ema_momentum, patch_size, pixel_weight, norm_pix):
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
            repr_loss, tgt_std = representation_loss(out, target_repr)

            if pixel_weight > 0.0:
                pix_loss = rvm_loss(
                    out,
                    batch["target"],
                    patch_size,
                    masked_only=True,
                    norm_pix=norm_pix,
                )
            else:
                pix_loss = jnp.zeros((), jnp.float32)

            total = repr_loss + pixel_weight * pix_loss
            return total, (repr_loss, pix_loss, tgt_std)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        repr_loss, pix_loss, tgt_std = aux

        state = state.apply_gradients(grads=grads)
        state = state.replace(
            ema_params=ema_update(state.ema_params, state.params, ema_momentum)
        )

        metrics = {
            "loss": loss,
            "repr_loss": repr_loss,
            "pixel_loss": pix_loss,
            "target_std": tgt_std,
            "grad_norm": optax.global_norm(grads),
        }
        return state, metrics, next_rng

    return jax.jit(train_step)


def make_eval_step(model, patch_size, pixel_weight, norm_pix):
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
        repr_loss, tgt_std = representation_loss(out, target_repr)

        if pixel_weight > 0.0:
            pix_loss = rvm_loss(
                out, batch["target"], patch_size, masked_only=True, norm_pix=norm_pix
            )
        else:
            pix_loss = jnp.zeros((), jnp.float32)

        return {
            "loss": repr_loss + pixel_weight * pix_loss,
            "repr_loss": repr_loss,
            "pixel_loss": pix_loss,
            "target_std": tgt_std,
        }

    return jax.jit(eval_step)


# --------------------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------------------


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
        self.patch_size = tuple(model_config.patch_size[-2:])

        self.pixel_weight = float(getattr(self.config, "pixel_weight", 1.0))
        self.norm_pix = bool(getattr(self.config, "norm_pix", False))

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

    # -- setup -------------------------------------------------------------------------

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

        # EMA target encoder starts as an exact copy of the (pretrained) params.
        ema_params = jax.tree_util.tree_map(jnp.array, params)

        steps_per_epoch = len(self.train_loader)
        tx, self.schedule = build_optimizer(self.config, steps_per_epoch)
        self.state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx, ema_params=ema_params
        )

        self.train_step_fn = make_train_step(
            self.model,
            float(self.config.ema_momentum),
            self.patch_size,
            self.pixel_weight,
            self.norm_pix,
        )
        self.eval_step_fn = make_eval_step(
            self.model, self.patch_size, self.pixel_weight, self.norm_pix
        )

        wandb_config = dataclasses.asdict(self.config) | {
            **dataclasses.asdict(self.model_config),
            "dtype": str(self.model_config.dtype),
        }
        wandb.init(
            project=self.config.wandb_project, name=self.config.wandb_name, config=wandb_config
        )

    def _load_and_merge(self, init_params):
        """Fill from the pretrained checkpoint; `decoder_proj` stays randomly initialized."""
        pretrained = load_pretrained_params(self.config.init_params_path)
        merged = merge_params(init_params, pretrained)

        ckpt_keys = set(_leaf_paths(pretrained))
        model_keys = set(_leaf_paths(merged))

        orphaned = ckpt_keys - model_keys
        if orphaned:
            raise ValueError(
                f"checkpoint leaves unused by the model -- build_model has drifted from "
                f"the checkpoint architecture: {sorted(orphaned)}"
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

    # -- training ----------------------------------------------------------------------

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
                        f"loss {metrics['loss']:.4f} repr {metrics['repr_loss']:.4f} "
                        f"pix {metrics['pixel_loss']:.4f} tgt_std {metrics['target_std']:.4f} "
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

    # -- evaluation --------------------------------------------------------------------

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
        self._save_reconstruction_visualization()

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

    def _save_reconstruction_visualization(self):
        """Ground-truth frame next to its full reconstruction, side by side.

        `out['reconstructed']` is already pixel-scale (the `Detokenizer` decodes it
        directly), so there's no unpatchify step -- except when `norm_pix` is on, where
        the head regresses normalized patches and we undo that with the ground-truth
        patch statistics before viewing it as pixels.
        """
        if self.eval_loader is None:
            return

        batch = _to_numpy_batch(next(iter(self.eval_loader)))
        rng = jax.random.PRNGKey(0)
        mask_rng, state_rng = jax.random.split(rng)
        out = self.model.apply(
            {"params": self.state.params},
            batch["source"],
            batch["target"],
            batch["target_deltas"],
            rng_key=mask_rng,
            rngs={"default": state_rng},
            method=self.model.reconstruct,
        )

        gt_img = jnp.asarray(batch["target"][0:1, 0:1])  # (1, 1, H, W, 3)
        pred_img = out["reconstructed"][0:1, 0:1].astype(jnp.float32)

        if self.norm_pix:
            gt_patches = patchify(gt_img, self.patch_size)
            mu = gt_patches.mean(axis=-1, keepdims=True)
            var = gt_patches.var(axis=-1, keepdims=True)
            pred_patches = patchify(pred_img, self.patch_size)
            pred_patches = pred_patches * jnp.sqrt(var + 1e-6) + mu
            pred_img = unpatchify(pred_patches, self.patch_size)

        grid = self._make_side_by_side(
            self._array_to_pil(np.asarray(gt_img[0, 0])),
            self._array_to_pil(np.asarray(pred_img[0, 0])),
        )

        vis_dir = os.path.join(self.checkpoint_dir, "image_vis")
        os.makedirs(vis_dir, exist_ok=True)
        grid.save(os.path.join(vis_dir, f"{self.epoch}_{self.global_step}.png"))

    @staticmethod
    def _array_to_pil(img: np.ndarray) -> Image.Image:
        arr = np.clip(img, 0, 1) * 255.0
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    @staticmethod
    def _make_side_by_side(left: Image.Image, right: Image.Image, pad: int = 4) -> Image.Image:
        width = left.width + right.width + 3 * pad
        height = max(left.height, right.height) + 2 * pad
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        canvas.paste(left, (pad, pad))
        canvas.paste(right, (2 * pad + left.width, pad))
        return canvas