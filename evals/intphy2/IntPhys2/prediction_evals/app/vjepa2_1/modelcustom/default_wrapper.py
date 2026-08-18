"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

modelcustom API requirements:

API requirements for Encoder module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])

API requirements for Predictor module:
    1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip tokens (shape=[batch_size x num_encoder_tokens x feature_dim])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])
    2) Needs to have a public attribute called 'embed_dim' (int) describing its
        output feature dimension.
"""

import logging
import os
import sys

import torch
import torch.nn.functional as F

# app/vjepa2_1/*.py use bare imports (e.g. "import vision_transformer", "from utils.mask_utils
# import ..."), so the vjepa2_1 package directory itself has to be on sys.path -- it is not meant
# to be imported as "app.vjepa2_1.make_model".
_VJEPA2_1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _VJEPA2_1_DIR not in sys.path:
    sys.path.insert(0, _VJEPA2_1_DIR)

import make_model  # noqa: E402

from src.masks.utils import apply_masks
from src.models.utils.multimask import MultiMaskWrapper

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Overrides applied on top of make_model._make_vjepa2_1_model's defaults to reproduce
# make_model.vjepa2_1_vit_giant_384's behavior without hitting its hardcoded img_size=384
# kwarg (calling that factory directly with an img_size override raises a duplicate-kwarg
# TypeError since it also forwards **kwargs into _make_vjepa2_1_model).
DEFAULT_BUILD_KWARGS = dict(
    arch_name="vit_giant_xformers",
    predictor_num_mask_tokens=8,
    return_all_tokens=True,
)


def init_module(
    frames_per_clip: int,
    nb_context_frames: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    model_kwargs = dict(model_kwargs)
    resolution = model_kwargs.pop("resolution", 384)

    build_kwargs = dict(DEFAULT_BUILD_KWARGS)
    build_kwargs.update(model_kwargs)

    pretrained = bool(checkpoint) and (os.path.isfile(checkpoint) or str(checkpoint).startswith("http"))
    if checkpoint and not pretrained:
        logger.warning(f"checkpoint '{checkpoint}' is not a local file or URL -- building with random init")

    encoder, predictor = make_model._make_vjepa2_1_model(
        model_name=checkpoint,
        pretrained=pretrained,
        img_size=resolution,
        num_frames=frames_per_clip,
        **build_kwargs,
    )

    # The predictor's I/O projections (predictor_embed / predictor_proj) are sized for the
    # concatenation of `n_output_distillation` hierarchical layers, not a single final-layer
    # embedding. Both the context tokens fed into the predictor and the targets it is compared
    # against therefore need to come from the hierarchical (multi-layer) encoder output.
    encoder.return_hierarchical = True

    encoder = MultiMaskWrapper(encoder)
    # There is a single pretrained encoder checkpoint (checkpoint_key, default "target_encoder"),
    # so it is reused for both context extraction and target extraction.
    target_encoder = encoder

    print(encoder)
    print(predictor)

    grid_size = resolution // encoder.backbone.patch_size
    grid_depth = frames_per_clip // encoder.backbone.tubelet_size
    model = AnticipativeWrapperNoAR(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        frames_per_clip=frames_per_clip,
        nb_context_frames=nb_context_frames,
        resolution=resolution,
        grid_size=grid_size,
        grid_depth=grid_depth,
        **wrapper_kwargs,
    )
    model.embed_dim = encoder.backbone.embed_dim

    return model


class AnticipativeWrapperNoAR(torch.nn.Module):
    """ Use predictor for inference """

    def __init__(
        self,
        encoder,
        target_encoder,
        predictor,
        frames_per_clip=16,
        nb_context_frames=5,
        resolution=384,
        no_predictor=False,
        grid_size=16,
        grid_depth=8,
        padding_type="zero",
    ):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.frames_per_clip = frames_per_clip
        self.nb_context_frames = nb_context_frames
        self.resolution = resolution
        self.grid_size = grid_size
        self.grid_depth = grid_depth

    def forward(self, x):
        """
        :param x: (Tensor) video of shape [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape

        # ----------------------------------------------------------------------- #
        # Compute Masks
        # ----------------------------------------------------------------------- #

        patch_size = self.encoder.backbone.patch_size
        tubelet_size = self.encoder.backbone.tubelet_size

        m, m_, full_m = get_time_masks(
            self.nb_context_frames,
            spatial_size=(patch_size, patch_size),
            temporal_size=tubelet_size,
            spatial_dim=(self.resolution, self.resolution),
            temporal_dim=self.frames_per_clip,
            as_bool=False,
        )
        full_m = full_m.unsqueeze(0).to(x.device)
        m = m.unsqueeze(0).to(x.device)
        m_ = m_.unsqueeze(0).to(x.device)

        masks_enc = [m.repeat(B, 1)]
        masks_pred = [m_.repeat(B, 1)]
        full_mask = [full_m.repeat(B, 1)]

        # ----------------------------------------------------------------------- #
        # Compute Targets
        # ----------------------------------------------------------------------- #
        h = self.target_encoder(x, full_mask)[0]
        # -- create targets (masked regions of h)
        targets = apply_masks(h, masks_pred, concat=False)

        # ----------------------------------------------------------------------- #
        # Compute Predictions
        # ----------------------------------------------------------------------- #
        context = self.encoder(x, masks_enc)[0]
        preds, _ = self.predictor(context, masks_enc[0], masks_pred[0], mod="video", mask_index=0)

        targets = targets[0]

        targets = F.layer_norm(targets, (targets.size(-1),))  # normalize over feature-dim  [B, N, D]

        return preds, targets


def get_time_masks(n_timesteps, spatial_size=(16, 16), temporal_size=2, spatial_dim=(224, 224), temporal_dim=16, as_bool=False):
    assert n_timesteps % temporal_size == 0
    x, y = spatial_dim
    t = temporal_dim

    num_patches_spatial = x / spatial_size[0] * y / spatial_size[1]
    num_patches_time = t / temporal_size
    patches_n_timesteps = int(num_patches_spatial * n_timesteps // temporal_size)

    patch_idcs = torch.arange(start=0, end=int(num_patches_spatial * num_patches_time), dtype=int)
    if as_bool:
        mask_enc = patch_idcs < patches_n_timesteps
        mask_pred = patch_idcs >= patches_n_timesteps

        full_mask = patch_idcs >= 0
    else:
        mask_enc = patch_idcs[:patches_n_timesteps]
        mask_pred = patch_idcs[patches_n_timesteps:]

        full_mask = patch_idcs

    return mask_enc, mask_pred, full_mask
