"""Causal, fully-masked pixel-reconstruction trainer for `VideoSiamMAE.reconstruct_tf`
(JAX + Flax). Every target frame `t` is predicted purely from the recurrent state built
after frame `t - 1` -- see `rvm_jax.py::VideoSiamMAE.reconstruct_tf` and
`rvm_tf.py::RecurrentWorldModel`'s `context_mode='state'` path, which this mirrors.

Unlike `optimier_ema_jax.py` (data2vec-style representation loss, EMA teacher), the loss
here is plain pixel-space `rvm_loss`, so there is no teacher network and no EMA -- this
file reuses only the generic, loss-agnostic infra (logging, checkpointing, the AdamW +
warmup-cosine optimizer) from `optimier_ema_jax.py`.

Run directly: `python src/optimisation/trainer_causal_jax.py --config path/to.yaml`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import yaml
from flax.training import train_state
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.rvm_dataset_tf import RVMDataset
from src.models.rvm_jax import build_model_L, context_target_split, rvm_loss
from src.optimisation.optimier_ema_jax import (
    build_optimizer,
    due,
    get_logger,
    load_and_merge_params,
    load_pytree_npz,
    save_pytree_npz,
)

CKPT_RE = re.compile(r"model_(\d+)_(\d+)\.npz")

@dataclasses.dataclass
class CausalTrainerConfig:
    """Hyperparameters for `Trainer`. Field names mirror `EMATrainerConfig` in
    `optimier_ema_jax.py` where the same optimizer/schedule machinery is reused; there
    are no EMA fields since this pipeline has no teacher network.
    """

    # runtime
    checkpoint_dir: str = "checkpoints_causal_jax"
    seed: Optional[int] = None
    init_params_path: Optional[str] = None  # .npz of pretrained params (decoder_embedder
    # etc. transfer directly from the MAE checkpoint; only `mask_token`-adjacent shapes
    # would need to match)

    # budget: epochs is a full pass over train_loader; total_steps is derived, not set
    epochs: int = 4
    total_steps: int = 0
    grad_accum: int = 1

    # optimizer -- AdamW with the betas/wd used by MAE-style ViT recipes
    lr_peak: float = 5.0e-5
    lr_min: float = 5.0e-7
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    weight_decay: float = 0.05
    wd_skip: Tuple[str, ...] = ("bias", "norm", "cls_token", "mask_token")
    grad_clip: Optional[float] = 1.0
    warmup_ratio: float = 0.05

    # causal split: frames [0, num_context) seed the state, [num_context, roll_out) are
    # the targets that get reconstructed (see `context_target_split`)
    num_context: int = 16 #! inspired from vjepa.21

    # loss -- kwargs forwarded to rvm_loss(); every target patch is masked by
    # construction, so masked_only is always False
    norm_pix: bool = False

    # misc
    log_interval: int = 50
    eval_interval: int = 1000

    # wandb
    wandb_project: str = "rwm"
    wandb_name: Optional[str] = None




def numpy_collate(batch):
    """`RVMDataset` (rvm_dataset_tf) items -> NHWC numpy, ready for `model.apply`."""
    context = np.stack(
        [item["context"].permute(0, 2, 3, 1).numpy() for item in batch], axis=0
    )  # (B, N, H, W, 3)
    frame_times = np.stack(
        [item["sampled_indices"].numpy() for item in batch], axis=0
    ).astype(np.float32)  # (B, N)
    return {"context": context, "frame_times": frame_times}


def build_dataloader(dataset_config, dataloader_config, shuffle: bool, num_workers=None):
    if dataset_config is None:
        return None
    dataset = RVMDataset(dataset_config, deterministic=not shuffle)
    if num_workers is None:
        num_workers = dataloader_config.get("num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=dataloader_config.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
        collate_fn=numpy_collate,
        persistent_workers=num_workers > 0,
    )




def make_train_step(model, config: CausalTrainerConfig, schedule, target_indices, patch_size):
    def train_step(state, batch, rng):
        step_rng, next_rng = jax.random.split(rng)
        target_frames = batch["context"][:, target_indices]  # (B, Tt, H, W, 3)

        def loss_fn(params):
            out = model.apply(
                {"params": params},
                batch["context"],
                target_indices,
                batch["frame_times"],
                rngs={"default": step_rng},
                method=model.reconstruct_tf,
            )
            return rvm_loss(
                out, target_frames, patch_size, masked_only=False, norm_pix=config.norm_pix
            )

        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        # capture the pre-update counter: apply_gradients advances `gradient_step`, so
        # reading it afterwards reports the LR of the *next* update
        applied_step = state.opt_state.gradient_step
        state = state.apply_gradients(grads=grads)
        did_update = state.opt_state.mini_step == 0

        metrics = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "lr": schedule(applied_step),
            "did_update": did_update.astype(jnp.float32),
        }
        return state, metrics, next_rng

    return jax.jit(train_step)


def make_eval_step(model, target_indices, patch_size):
    def eval_step(params, batch, rng):
        out = model.apply(
            {"params": params},
            batch["context"],
            target_indices,
            batch["frame_times"],
            rngs={"default": rng},
            method=model.reconstruct_tf,
        )
        target_frames = batch["context"][:, target_indices]
        loss = rvm_loss(out, target_frames, patch_size, masked_only=False)
        return {"loss": loss}

    return jax.jit(eval_step)


class Trainer:
    def __init__(
        self,
        train_loader,
        eval_loader=None,
        config: Optional[CausalTrainerConfig] = None,
    ):
        self.config = config or CausalTrainerConfig()
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.epochs = int(self.config.epochs)
        self.current_epoch = 0
        self.global_step = 0

        self.model = build_model_L(masking_ratio=1.0)  # masking_ratio is unused by
        # reconstruct_tf (masking is 100% by construction); kept at 1.0 for honesty

        self.checkpoint_dir = os.path.join(
            self.config.checkpoint_dir, f"exp_{int(time.time())}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, "training.log"))

        self.max_checkpoints_to_keep = 3
        self.rng = jax.random.PRNGKey(self.config.seed or 0)

        # -- keep total_steps (which drives the LR schedule) equal to the number of
        # batches the loop will actually run, or the cosine decay finishes early / never
        # finishes, with nothing to indicate it.
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
                    f"using {derived} so the schedule spans the real run.",
                    stacklevel=2,
                )
            if configured != derived:
                self.config = dataclasses.replace(self.config, total_steps=derived)

        self._init_state()

    def _init_state(self):
        self.rng, init_rng, state_rng = jax.random.split(self.rng, 3)
        first_batch = next(iter(self.train_loader))

        num_frames = first_batch["context"].shape[1]
        assert 1 <= self.config.num_context < num_frames, (
            f"num_context={self.config.num_context} must leave at least one target "
            f"frame out of {num_frames}"
        )
        self.context_indices, self.target_indices = context_target_split(
            num_frames, self.config.num_context
        )
        self.patch_size = tuple(self.model.detokenizer.patch_size)

        params = self.model.init(
            {"params": init_rng, "default": state_rng},
            first_batch["context"],
            self.target_indices,
            first_batch["frame_times"],
            method=self.model.reconstruct_tf,
        )["params"]

        if self.config.init_params_path is not None:
            params = load_and_merge_params(params, self.config.init_params_path)

        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        self.logger.info(f"model params: {n_params / 1e6:.1f}M")

        tx, self.schedule = build_optimizer(self.config, params)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

        self.train_step_fn = make_train_step(
            self.model, self.config, self.schedule, self.target_indices, self.patch_size
        )
        self.eval_step_fn = make_eval_step(self.model, self.target_indices, self.patch_size)

        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            config=dataclasses.asdict(self.config),
        )
        # one x-axis for everything: wandb requires non-decreasing steps, so mixing
        # auto-increment / step=epoch / step=global_step drops points silently
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")


    def _checkpoints(self) -> list[tuple[int, int, str]]:
        """(epoch, step, path) for every checkpoint in `checkpoint_dir`, oldest first."""
        found = []
        for fname in os.listdir(self.checkpoint_dir):
            m = CKPT_RE.fullmatch(fname)
            if m:
                found.append(
                    (int(m.group(1)), int(m.group(2)), os.path.join(self.checkpoint_dir, fname))
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
            params=restored["params"], opt_state=restored["opt_state"], step=restored["step"]
        )
        self.global_step = int(step)
        # mid-epoch checkpoints replay their epoch from the start (the loader can't be
        # positioned mid-stream); end-of-epoch ones move on
        self.current_epoch = int(epoch) + int(restored.get("epoch_done", 0))
        if "rng" in restored:
            self.rng = jnp.asarray(restored["rng"], dtype=jnp.uint32)
        self.logger.info(f"resumed from {path} (epoch {self.current_epoch}, step {self.global_step})")
        return True


    def _log(self, payload: dict):
        wandb.log(payload | {"epoch": self.current_epoch, "global_step": self.global_step})

    def eval(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        n_batches = 0
        # fixed key on purpose: eval should be deterministic across evaluations so the
        # curve reflects the model, not the draw
        rng = jax.random.PRNGKey(0)
        for batch in tqdm(self.eval_loader, total=len(self.eval_loader)):
            rng, step_rng = jax.random.split(rng)
            metrics = self.eval_step_fn(self.state.params, batch, step_rng)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    def _eval_and_save(self, epoch_done: bool = False) -> str:
        """Every checkpoint is written at an eval boundary, so model_{epoch}_{step}
        always lines up with a logged eval point."""
        if self.eval_loader is not None:
            metrics = self.eval()
            self.logger.info(
                f"epoch {self.current_epoch} step {self.global_step} eval_loss {metrics['loss']:.4f}"
            )
            self._log({f"eval/{k}": v for k, v in metrics.items()})
        path = self.save_checkpoint(epoch_done=epoch_done)
        self.logger.info(f"saved {path}")
        return path


    def train(self, max_steps: Optional[int] = None, resume: bool = False):
        """Run the training loop.

        Args:
          max_steps: optional early stop after this many batches. This does *not*
            rescale the LR schedule -- it is a debug cap, not a shorter run.
          resume: pick up from the newest checkpoint in `checkpoint_dir` if one exists.
        """
        if resume:
            self.resume()

        stop = False
        for epoch in tqdm(
            range(self.current_epoch, self.epochs), desc="training",
            initial=self.current_epoch, total=self.epochs,
        ):
            self.current_epoch = epoch
            batches = tqdm(
                self.train_loader, desc=f"epoch {epoch}", total=self.steps_per_epoch, leave=False
            )
            for batch in batches:
                self.state, metrics, self.rng = self.train_step_fn(self.state, batch, self.rng)
                self.global_step = int(self.state.step)

                if due(self.global_step, self.config.log_interval):
                    m = {k: float(v) for k, v in metrics.items()}
                    self.logger.info(
                        f"epoch {epoch} step {self.global_step} loss {m['loss']:.4f} lr {m['lr']:.2e}"
                    )
                    self._log({f"train/{k}": v for k, v in m.items()})

                if due(self.global_step, self.config.eval_interval):
                    self._eval_and_save()

                if max_steps is not None and self.global_step >= int(max_steps):
                    stop = True
                    break

            batches.close()
            self._eval_and_save(epoch_done=True)

            if stop:
                self.logger.info(f"stopping early at step {self.global_step} (max_steps)")
                break

        wandb.finish()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_jax_causal.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if config.get("seed") is not None:
        np.random.seed(config["seed"])

    dataloader_config = config["dataloader"]
    train_loader = build_dataloader(config["dataset"]["train"], dataloader_config, shuffle=True)
    eval_loader = build_dataloader(
        config["dataset"].get("eval"), dataloader_config, shuffle=False,
        num_workers=dataloader_config.get("eval_num_workers", min(4, dataloader_config.get("num_workers", 4))),
    )

    trainer_config = CausalTrainerConfig(**config.get("trainer", {}))
    trainer = Trainer(train_loader, eval_loader, trainer_config)
    trainer.train(max_steps=args.max_steps, resume=args.resume)


if __name__ == "__main__":
    main()
