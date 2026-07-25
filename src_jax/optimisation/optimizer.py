import dataclasses
import os
import time
from collections.abc import Mapping
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax.training import train_state
from PIL import Image
from tqdm import tqdm

from src_jax.models.rvm_jax import RVMConfig, build_model, patchify, unpatchify, rvm_loss
from src_jax.optimisation.config import TrainerConfig


def _to_numpy_batch(batch):
    """Torch batch (from `RVMDataset.collate_fn`, NHWC) -> numpy, ready for `model.apply`."""
    return {
        "source": batch["source"].numpy(),
        "target": batch["target"].numpy(),
        "target_deltas": batch["target_deltas"].numpy().astype(np.int32),
    }


def _tree_shapes(tree) -> dict:
    """Plain-dict-of-shapes view of a params pytree, robust to dict vs. FrozenDict nodes."""
    if isinstance(tree, Mapping):
        return {k: _tree_shapes(v) for k, v in tree.items()}
    return tuple(jnp.shape(tree))


def load_pretrained_params(path: str) -> dict:
    """Load a Flax params pytree saved as a flat "/"-joined-key .npz (see recover_tree)."""
    flat = np.load(path, allow_pickle=False)
    tree = {}
    for key in flat.files:
        parts = key.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = jnp.asarray(flat[key])
    return tree


def build_schedule(config: TrainerConfig, steps_per_epoch: int) -> optax.Schedule:
    """Linear warmup -> cosine decay to `min_lr`, matching `src/optimisation/optimizer.py`."""
    total_steps = max(steps_per_epoch * int(config.num_epochs), 1)
    warmup_steps = steps_per_epoch * int(config.warmup_epochs)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(config.lr),
        warmup_steps=warmup_steps,
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


def make_train_step(model, patch_size, masked_only, norm_pix):
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
            loss = rvm_loss(out, batch["target"], patch_size, masked_only, norm_pix)
            return loss, out

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, next_rng

    return jax.jit(train_step)


def make_eval_step(model, patch_size, masked_only, norm_pix):
    def eval_step(params, batch, rng):
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
        return rvm_loss(out, batch["target"], patch_size, masked_only, norm_pix)

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
        self.patch_size = tuple(model_config.patch_size[-2:])

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
            init_shapes = _tree_shapes(params)
            pretrained_shapes = _tree_shapes(pretrained)
            if init_shapes != pretrained_shapes:
                raise ValueError(
                    f"pretrained params at {self.config.init_params_path} don't match "
                    f"model_config's param tree (mismatched keys/shapes):\n"
                    f"expected: {init_shapes}\ngot: {pretrained_shapes}"
                )
            params = pretrained
            print(f"initialized params from {self.config.init_params_path}")

        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"model params: {n_params / 1e6:.1f}M")

        steps_per_epoch = len(self.train_loader)
        #! check this
        tx, self.schedule = build_optimizer(self.config, steps_per_epoch)
        # in jax we need a train state I belive.
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

        self.train_step_fn = make_train_step(
            self.model, self.patch_size, self.config.masked_only, self.config.norm_pix
        )
        self.eval_step_fn = make_eval_step(
            self.model, self.patch_size, self.config.masked_only, self.config.norm_pix
        )

        wandb_config = dataclasses.asdict(self.config) | {
            **dataclasses.asdict(self.model_config),
            "dtype": str(self.model_config.dtype),
        }
        wandb.init(
            project=self.config.wandb_project, name=self.config.wandb_name, config=wandb_config
        )

    def save_checkpoint(self):
        ckpt = {
            "params": self.state.params,
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
            "opt_state": self.state.opt_state,
            "step": self.state.step,
            "epoch": self.epoch,
        }
        restored = self.ckpt_mgr.restore(latest, args=ocp.args.StandardRestore(target))
        self.state = self.state.replace(
            params=restored["params"], opt_state=restored["opt_state"], step=restored["step"]
        )
        self.epoch = int(restored["epoch"])
        self.global_step = int(latest)
        return True

    def train(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or int(self.config.num_epochs)
        for _ in tqdm(range(num_epochs), total=num_epochs, leave=False):
            for batch in tqdm(self.train_loader, total=len(self.train_loader), leave=True):
                loss = self._train_step(batch)
                self.global_step += 1
                if self.global_step % int(self.config.log_interval) == 0:
                    lr = float(self.schedule(self.state.step))
                    print(f"epoch {self.epoch} step {self.global_step} loss {loss:.4f} lr {lr:.2e}")
                    wandb.log(
                        {"train/loss": loss, "train/lr": lr, "epoch": self.epoch},
                        step=self.global_step,
                    )

                if self.eval_loader is not None and self.global_step % int(self.config.eval_interval) == 0:
                    self._evaluate_and_log()

            self.epoch += 1
            self.save_checkpoint()

            if self.eval_loader is not None:
                self._evaluate_and_log()

    def _train_step(self, batch):
        batch = _to_numpy_batch(batch)
        self.state, loss, self.rng = self.train_step_fn(self.state, batch, self.rng)
        return float(loss)

    def _evaluate_and_log(self):
        eval_loss = self.eval()
        print(f"epoch {self.epoch} step {self.global_step} eval_loss {eval_loss:.4f}")
        wandb.log({"eval/loss": eval_loss, "epoch": self.epoch}, step=self.global_step)
        self._save_reconstruction_visualization()

    def eval(self):
        total_loss, n_batches = 0.0, 0
        rng = jax.random.PRNGKey(0)
        for batch in self.eval_loader:
            batch = _to_numpy_batch(batch)
            rng, step_rng = jax.random.split(rng)
            loss = self.eval_step_fn(self.state.params, batch, step_rng)
            total_loss += float(loss)
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _save_reconstruction_visualization(self):
        """Ground-truth frame next to its full reconstruction, side by side.

        `out['reconstructed']` is already pixel-scale (the `Detokenizer` decodes it
        directly), so unlike the PyTorch trainer there's no unpatchify step -- except
        when `norm_pix` is on, where the head regresses normalized patches and we have
        to undo that with the ground-truth patch stats before viewing it as pixels.
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

        gt_img = jnp.asarray(batch["target"][0:1, 0:1])           # (1, 1, H, W, 3)
        pred_img = out["reconstructed"][0:1, 0:1].astype(jnp.float32)

        if self.config.norm_pix:
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
