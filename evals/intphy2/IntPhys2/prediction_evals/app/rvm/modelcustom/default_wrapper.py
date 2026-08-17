"""
------------------------------------------------------------------------------

modelcustom API requirements (see e.g. `app/videomaev2/modelcustom/default_wrapper.py`):

    A pytorch `nn.Module` with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (preds, targets), each (Tensor) of shape [batch_size x num_output_tokens x feature_dim]

RVM (`app/rvm/rvm_jax.py`) is a JAX/Flax model, unlike the other (native torch) models in
this eval harness. This wrapper is a plain `nn.Module` with no torch parameters -- it just
holds the (immutable) Flax module + its restored params and dispatches to jax under the hood,
converting to/from torch tensors at the `forward()` boundary so it satisfies the same contract
as every other wrapper here. `evals/intphys2/eval.py` imports torch before dynamically importing
this module, so jax loads second -- the required order (see `ablations/ssv2_probe/trainer_jax.py`
for why: torch/triton and jaxlib each bundle their own LLVM, and whichever loads second wins).

RVM reconstructs future frames directly in pixel space (it has no target/EMA encoder), so -- like
`app/videomaev2/modelcustom/default_wrapper.py` -- `preds`/`targets` here are patchified pixels,
not learned representations.
"""

import logging

import einops
import jax
import jax.numpy as jnp
import numpy as np
import torch

import app.rvm.rvm_jax as rvm_jax

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _recover_tree(flat_dict):
    """Un-flatten a `{'a/b/c': array}` dict (as saved in the restored npz) into the
    nested dict pytree flax expects for `apply({'params': ...})`."""
    tree = {}
    for k, v in flat_dict.items():
        parts = k.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v
    return tree


def _build_reconstruct_fn(rvm_model):
    """Jitted, pure wrapper around the frozen RVM's `reconstruct` method.

    `rvm_model` is closed over (a flax Module is a frozen dataclass, not a jit argument),
    so the only traced inputs are params/source/target/deltas/rng.
    """

    def _forward(params, source, target, deltas, rng):
        out = rvm_model.apply(
            {"params": params},
            source, target, deltas,
            method=rvm_model.reconstruct,
            rngs={"default": rng},
        )
        return out["reconstructed"].astype(jnp.float32)  # (B, Tt, H, W, 3)

    return jax.jit(_forward)


def init_module(
    frames_per_clip: int,
    nb_context_frames: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    model_kwargs = model_kwargs or {}
    wrapper_kwargs = dict(wrapper_kwargs or {})

    variant = model_kwargs.get("variant", "L")
    logger.info(f"Building RVM (variant={variant})")
    if variant == "L":
        rvm_model = rvm_jax.build_model_L()
    elif variant == "S":
        rvm_model = rvm_jax.build_model(rvm_jax.RVMConfig())
    else:
        raise ValueError(f"Unknown RVM variant {variant!r}, expected 'L' or 'S'.")

    logger.info(f"Loading pretrained RVM weights from {checkpoint}")
    restored = _recover_tree(dict(np.load(checkpoint, allow_pickle=False)))
    params = jax.tree_util.tree_map(jnp.asarray, restored)
    count = sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(params))
    logger.info(f"RVM params: {count:,}")

    reconstruct_fn = _build_reconstruct_fn(rvm_model)

    model = AnticipativeWrapperNoAR(
        rvm_model=rvm_model,
        params=params,
        reconstruct_fn=reconstruct_fn,
        frames_per_clip=frames_per_clip,
        nb_context_frames=nb_context_frames,
        **wrapper_kwargs,
    )
    return model


class AnticipativeWrapperNoAR(torch.nn.Module):
    """Rolls RVM's recurrent encoder over the context frames, then reconstructs the
    remaining (target) frames in pixel space, VideoMAE-style."""

    def __init__(
        self,
        rvm_model,
        params,
        reconstruct_fn,
        frames_per_clip=16,
        nb_context_frames=4,
        frame_step=1,
        seed=0,
    ):
        super().__init__()
        self.rvm_model = rvm_model
        self.params = params
        self._reconstruct_fn = reconstruct_fn
        self.frames_per_clip = frames_per_clip
        self.nb_context_frames = nb_context_frames
        # Raw video frame stride between consecutive clip frames -- needed to convert a
        # target frame's position in the clip into `target_deltas` (a frame gap in raw-video
        # units, as at train time; see `src/datasets/rvm_dataset.py`). Unlike
        # `nb_context_frames`/`frames_per_clip`, `evals/intphys2/eval.py` never mutates this
        # on the model, so it must match the `frame_steps` value in the eval config.
        self.frame_step = frame_step
        self.patch_size = (16, 16)
        self._rng_key = jax.random.PRNGKey(seed)

    def forward(self, x):
        """
        :param x: (Tensor) video of shape [B, C, T, H, W], ImageNet-normalized.
        """
        B, C, T, H, W = x.shape
        device = x.device

        Ts = self.nb_context_frames
        Tt = self.frames_per_clip - Ts
        assert T == self.frames_per_clip, f"expected {self.frames_per_clip} frames, got {T}"
        assert Tt > 0, "no target frames left to predict"

        # ----------------------------------------------------------------------- #
        # Undo ImageNet normalization -> raw [0, 1] pixels (RVM's native input space,
        # see `notebooks/rvm_inference_example.ipynb`: frames are fed as `frame / 255.0`)
        # ----------------------------------------------------------------------- #
        mean = torch.as_tensor(IMAGENET_MEAN, device=device, dtype=torch.float32)[None, :, None, None, None]
        std = torch.as_tensor(IMAGENET_STD, device=device, dtype=torch.float32)[None, :, None, None, None]
        pixels = x.float() * std + mean  # (B, C, T, H, W), in [0, 1]

        source = pixels[:, :, :Ts]  # (B, C, Ts, H, W)
        target = pixels[:, :, Ts:]  # (B, C, Tt, H, W)

        # channels-last numpy, as expected by the flax model
        source_np = source.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()
        target_np = target.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()

        # frame gap (in raw video frames) between the last context frame and each target
        # frame, matching how `target_deltas` was defined at train time.
        deltas_np = (self.frame_step * np.arange(1, Tt + 1))[None, :].repeat(B, axis=0).astype(np.int32)

        self._rng_key, step_key = jax.random.split(self._rng_key)
        reconstructed = self._reconstruct_fn(self.params, source_np, target_np, deltas_np, step_key)
        reconstructed = torch.from_numpy(np.asarray(reconstructed)).to(device)  # (B, Tt, H, W, 3)

        target_chw = target.permute(0, 2, 3, 4, 1).contiguous()  # (B, Tt, H, W, 3)

        # Patchify to (B, Tt*h*w, ph*pw*3) pixel tokens -- same convention as the videomaev2
        # wrapper, so the shared `F.l1_loss(...).mean((1, 2))` in `evals/intphys2/eval.py`
        # averages over comparable units (one pixel patch) across models.
        ph, pw = self.patch_size
        preds = einops.rearrange(
            reconstructed, "b t (H h) (W w) c -> b (t H W) (h w c)", h=ph, w=pw
        )
        targets = einops.rearrange(
            target_chw, "b t (H h) (W w) c -> b (t H W) (h w c)", h=ph, w=pw
        )

        return preds, targets
