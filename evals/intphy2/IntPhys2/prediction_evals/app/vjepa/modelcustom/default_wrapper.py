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
import copy
import torch.nn.functional as F

import app.vjepa.models.vision_transformer as vit
import app.vjepa.models.predictor as vit_pred
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.masks.utils import apply_masks
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
    try:
        checkpoint = torch.load(checkpoint, map_location="cpu")
        load = True
    except:
        load = False

    # ----------------------------------------------------------------------- #
    # Initialize Encoder
    # ----------------------------------------------------------------------- #
    
    resolution = model_kwargs["resolution"]
    enc_kwargs = model_kwargs["encoder"]
    enc_ckp_key = enc_kwargs.get("checkpoint_key")
    enc_model_name = enc_kwargs.get("model_name")

    encoder = vit.__dict__[enc_model_name](
            img_size=resolution,
            num_frames=frames_per_clip,
            **enc_kwargs
        )
    #Has to be done before the wrapper
    target_encoder = copy.deepcopy(encoder)
    encoder = MultiMaskWrapper(encoder)

    if load:
        pretrained_dict = checkpoint[enc_ckp_key]
        # --
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        #pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        for k, v in encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pretrained_dict[k] = v
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained model with msg: {msg}")
    print(encoder)

    # ----------------------------------------------------------------------- #
    # Initialize Target Encoder
    # ----------------------------------------------------------------------- #
    target_encoder = MultiMaskWrapper(target_encoder)

    target_enc_kwargs = model_kwargs["target_encoder"]
    target_enc_ckp_key = target_enc_kwargs.get("checkpoint_key")

    if load:
        pretrained_dict = checkpoint[target_enc_ckp_key]
        # --
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        #pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        for k, v in target_encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pretrained_dict[k] = v
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained model with msg: {msg}")

    # ----------------------------------------------------------------------- #
    # Initialize Predictor
    # ----------------------------------------------------------------------- #
    pred_kwargs = model_kwargs["predictor"]
    pred_ckp_key = pred_kwargs.get("checkpoint_key")
    pred_model_name = pred_kwargs.get("model_name")

    use_rope = 'rope' in enc_model_name
    rope_is_1D = 'rope1D' in enc_model_name
    predictor = vit_pred.__dict__[pred_model_name](
        img_size=resolution,
        num_frames=frames_per_clip,
        embed_dim=encoder.backbone.embed_dim,
        num_heads=encoder.backbone.num_heads,
        use_rope=use_rope,
        rope_is_1D=rope_is_1D,
        **pred_kwargs
        )
    predictor = PredictorMultiMaskWrapper(predictor)
    
    if load:
        pretrained_dict = checkpoint[pred_ckp_key]
        # --
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        #pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        for k, v in predictor.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pretrained_dict[k] = v
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained model with msg: {msg}")
    print(predictor)

    # ----------------------------------------------------------------------- #
    # Build Wrapper
    # ----------------------------------------------------------------------- #
    grid_size = resolution // encoder.backbone.patch_size
    grid_depth = frames_per_clip // encoder.backbone.tubelet_size
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
        
        # ----------------------------------------------------------------------- #
        # Compute Masks
        # ----------------------------------------------------------------------- #

        m,m_,full_m = get_time_masks(self.nb_context_frames,spatial_size=(self.encoder.backbone.patch_size,self.encoder.backbone.patch_size),temporal_dim=self.frames_per_clip,as_bool=False)
        full_m = full_m.unsqueeze(0).to(x.device)
        m = m.unsqueeze(0).to(x.device)
        m_ = m_.unsqueeze(0).to(x.device)

        masks_enc = [m.repeat(B, 1)]
        masks_pred = [m_.repeat(B, 1)]
        full_mask = [full_m.repeat(B, 1)]

        # ----------------------------------------------------------------------- #
        # Compute Targets
        # ----------------------------------------------------------------------- #
        h = self.target_encoder(x,full_mask)[0]
        # -- create targets (masked regions of h)
        targets = apply_masks(h, masks_pred, concat=False)


        # ----------------------------------------------------------------------- #
        # Compute Predictions
        # ----------------------------------------------------------------------- #
          

        context = self.encoder(x, masks_enc)
        preds = self.predictor(context, targets, masks_enc, masks_pred)

        preds = preds[0]
        targets = targets[0]

        targets = F.layer_norm(targets, (targets.size(-1),))  # normalize over feature-dim  [B, N, D]
        #preds = F.layer_norm(predictions, (predictions.size(-1),))

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