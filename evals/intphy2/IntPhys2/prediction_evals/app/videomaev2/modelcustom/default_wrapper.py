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

import torch

import app.videomaev2.model as models
import torch.nn.functional as F
import copy
from einops import rearrange


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    frames_per_clip: int,
    nb_context_frames: int, 
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    logger.info(f"Loading pretrained model from {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location="cpu")

    # ----------------------------------------------------------------------- #
    # Initialize Encoder
    # ----------------------------------------------------------------------- #
    resolution = model_kwargs["resolution"]
    enc_kwargs = model_kwargs["encoder"]
    enc_model_name = enc_kwargs.get("model_name")


    encoder = models.__dict__[enc_model_name]()
        
    pretrained_dict = checkpoint['model']
    # --
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained model with msg: {msg}")
    
    # ----------------------------------------------------------------------- #
    # Initialize Target Encoder
    # ----------------------------------------------------------------------- #
    target_encoder = None
    
    # ----------------------------------------------------------------------- #
    # Initialize Predictor
    # ----------------------------------------------------------------------- #
    predictor = None


    # ----------------------------------------------------------------------- #
    # Build Wrapper
    # ----------------------------------------------------------------------- #
    grid_size = resolution // 16
    grid_depth = frames_per_clip // 2
    model = AnticipativeWrapperNoAR(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        frames_per_clip=frames_per_clip,
        nb_context_frames=nb_context_frames,
        grid_size=grid_size,
        grid_depth=grid_depth,
        **wrapper_kwargs,
    )

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
        self.grid_size = grid_size
        self.grid_depth = grid_depth

    def forward(self, x):
        """
        :param x: (Tensor) video of shape [B, C, T, H, W]
        """
        B, C, T, H, W = x.shape
        device = x.device
        
        # ----------------------------------------------------------------------- #
        # Compute Masks
        # ----------------------------------------------------------------------- #

        m,m_,full_m = get_time_masks(self.nb_context_frames,spatial_size=(14,14),temporal_dim=self.frames_per_clip,as_bool=True)
        full_m = full_m.unsqueeze(0).to(device)
        m = m.unsqueeze(0).to(device)
        m_ = m_.unsqueeze(0).to(device)
        
        masks_enc = m.repeat(B, 1)
        masks_pred = m_.repeat(B, 1)
        full_mask = full_m.repeat(B, 1)

        # ----------------------------------------------------------------------- #
        # Compute Targets
        # ----------------------------------------------------------------------- #
        mean = torch.as_tensor((0.485, 0.456, 0.406)).to(device)[None, :, None, None, None]
        std = torch.as_tensor((0.229, 0.224, 0.225)).to(device)[None, :, None, None, None]
        unnorm_videos = x * std + mean  # in [0, 1]

        videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=14, p2=14)
        var = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
        mean = videos_squeeze.mean(dim=-2, keepdim=True)
        videos_norm = (videos_squeeze - mean) / (var)
        # we find that the mean is about 0.48 and standard deviation is about 0.08.
        videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
        B, _, C = videos_patch.shape
        #logger.info(videos_patch.shape)
        #logger.info(masks_pred.shape)
        targets = videos_patch[masks_pred].reshape(B, -1, C)


        # ----------------------------------------------------------------------- #
        # Compute Predictions
        # ----------------------------------------------------------------------- #
          

        preds = self.encoder(x,masks_pred,decoder_blocks=-1)
                  
        #preds = preds.view(num_videos,-1,*preds.shape[1:])
        #targets = targets.view(num_videos,-1,*targets.shape[1:])

        return preds, targets




def get_time_masks(n_timesteps,spatial_size=(16,16),temporal_size=2,spatial_dim=(224,224),temporal_dim=16,as_bool=False):
    assert n_timesteps % temporal_size == 0
    x,y = spatial_dim
    t = temporal_dim
    
    num_patches_spatial = x/spatial_size[0] * x/spatial_size[0]
    num_patches_time = t/temporal_size
    patches_n_timesteps = int(num_patches_spatial*n_timesteps//temporal_size)
    
    patch_idcs = torch.arange(start=0,end=int(num_patches_spatial*num_patches_time),dtype=int)
    if as_bool:
        mask_enc = patch_idcs < patches_n_timesteps
        mask_pred = patch_idcs >= patches_n_timesteps
    
        full_mask = patch_idcs >= 0
    else:
        mask_enc = patch_idcs[:patches_n_timesteps]
        mask_pred = patch_idcs[patches_n_timesteps:]
    
        full_mask = patch_idcs
    
    return mask_enc, mask_pred,full_mask