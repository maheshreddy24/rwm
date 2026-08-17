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
import os

import einops
import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
import torch.utils.dlpack

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


def _enable_compilation_cache():
    """Cache compiled XLA programs to disk so repeated process launches (re-runs while
    iterating, resuming after a crash, ...) don't re-pay compilation for shapes already
    seen. Must be set before the first `jax.jit` call actually compiles anything."""
    cache_dir = os.environ.get("JAX_RVM_CACHE_DIR", os.path.expanduser("~/.cache/jax_rvm_eval"))
    try:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        logger.info(f"JAX persistent compilation cache: {cache_dir}")
    except Exception as e:
        logger.warning(f"Could not enable JAX persistent compilation cache: {e}")


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

    _enable_compilation_cache()

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
        pad_batch_to=12,
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
        # `evals/intphys2/eval.py` batches pieces into chunks of its hardcoded
        # `CHUNK_SIZE = 12`; the trailing chunk of a video (and videos of a different
        # length) can be smaller. Every distinct batch size is a distinct `jax.jit` trace
        # -> a fresh XLA compile, which is what was making the GPU look "spiky"/idle
        # (compiling on the host) instead of just slow (running matmuls). Padding
        # multi-item batches up to a fixed size avoids that; single-item calls (the
        # `max_context_mode` smaller-context sweep, always batch size 1) are left alone
        # since they're already shape-stable across every video and every context length.
        self.pad_batch_to = pad_batch_to
        self.patch_size = (16, 16)
        self._rng_key = jax.random.PRNGKey(seed)

    def _pad_batch(self, t):
        """Pad batch dim 0 up to `self.pad_batch_to` by repeating the last sample.
        Returns (padded_tensor, padded_batch_size); caller slices the output back down."""
        b = t.shape[0]
        target_b = self.pad_batch_to
        if target_b is None or b == 1 or b >= target_b:
            return t, b
        pad = t[-1:].expand(target_b - b, *t.shape[1:])
        return torch.cat([t, pad], dim=0), target_b

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

        # channels-last, as expected by the flax model
        source = pixels[:, :, :Ts].permute(0, 2, 3, 4, 1).contiguous()  # (B, Ts, H, W, C)
        target = pixels[:, :, Ts:].permute(0, 2, 3, 4, 1).contiguous()  # (B, Tt, H, W, C)

        source_padded, pad_b = self._pad_batch(source)
        target_padded, _ = self._pad_batch(target)

        # Zero-copy GPU handoff to jax via dlpack -- avoids a blocking device->host->device
        # round trip (`.cpu().numpy()` / `torch.from_numpy`) on every single call, which was
        # stalling the GPU pipeline between every one of the many small forward() calls
        # `evals/intphys2/eval.py` makes per video.
        source_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(source_padded))
        target_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(target_padded))

        # frame gap (in raw video frames) between the last context frame and each target
        # frame, matching how `target_deltas` was defined at train time.
        deltas_jax = jnp.broadcast_to(
            jnp.arange(1, Tt + 1, dtype=jnp.int32) * self.frame_step, (pad_b, Tt)
        )

        self._rng_key, step_key = jax.random.split(self._rng_key)
        reconstructed = self._reconstruct_fn(self.params, source_jax, target_jax, deltas_jax, step_key)
        reconstructed = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(reconstructed))
        reconstructed = reconstructed[:B]  # drop the padded rows -- (B, Tt, H, W, 3)

        # Patchify to (B, Tt*h*w, ph*pw*3) pixel tokens -- same convention as the videomaev2
        # wrapper, so the shared `F.l1_loss(...).mean((1, 2))` in `evals/intphys2/eval.py`
        # averages over comparable units (one pixel patch) across models.
        ph, pw = self.patch_size
        preds = einops.rearrange(
            reconstructed, "b t (H h) (W w) c -> b (t H W) (h w c)", h=ph, w=pw
        )
        targets = einops.rearrange(
            target, "b t (H h) (W w) c -> b (t H W) (h w c)", h=ph, w=pw
        )

        return preds, targets
