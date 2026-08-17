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
import math
import torchvision

from argparse import Namespace

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from einops import rearrange

import sys
sys.path.insert(0, "/checkpoint/amaia/video/garridoq/code/cosmos-predict1")
from cosmos_predict1.autoregressive.inference.world_generation_pipeline import ARBaseLatentGenerationPipeline
from cosmos_predict1.autoregressive.utils.inference import validate_args
from cosmos_predict1.utils import log, misc





def init_module(
    frames_per_clip: int,
    nb_context_frames: int, 
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
):
    args = Namespace(
        input_type="video",
        ar_model_dir="Cosmos-Predict1-4B",
        top_p=0.9,
        temperature=0.0,
        num_input_frames=9,
        num_gpus=1,
        input_image_or_video_path="placeholder",
        video_save_folder="video_save_folder",
        checkpoint_dir=checkpoint,
        disable_diffusion_decoder=True,
        disable_guardrail=True,
        offload_guardrail_models=False,
        offload_diffusion_decoder=False,
        offload_ar_model=False,
        offload_tokenizer=False,
        batch_input_path=None,
        seed=0,
        data_resolution=[640, 1024],
    )

    inference_type = "base"  # When the inference_type is "base", AR model does not take text as input, the world generation is purely based on the input video
    
    sampling_config = validate_args(args, inference_type)

    # Initialize base generation model pipeline
    pipeline = ARBaseLatentGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.ar_model_dir,
        disable_diffusion_decoder=args.disable_diffusion_decoder,
        offload_guardrail_models=args.offload_guardrail_models,
        offload_diffusion_decoder=args.offload_diffusion_decoder,
        offload_network=args.offload_ar_model,
        offload_tokenizer=args.offload_tokenizer,
        disable_guardrail=args.disable_guardrail,
        parallel_size =1,
    )


     # ----------------------------------------------------------------------- #
    # Build Wrapper
    # ----------------------------------------------------------------------- #
    
    model = AnticipativeWrapperNoAR(
        pipeline=pipeline,
        args=args,
        sampling_config=sampling_config,
    )

    return model

def resize_input(video: torch.Tensor, resolution: list[int]):
    orig_h, orig_w = video.shape[-2], video.shape[-1]
    target_h, target_w = resolution
    logger.info(f"{orig_h=} {orig_w=}")
    logger.info(f"{video.shape=}")
    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    B, C, T, H, W = video.shape
    video_resized = torch.zeros((B, C, T, resizing_shape[0], resizing_shape[1]), dtype=video.dtype, device=video.device)
    for t in range(T):
        video_resized[:, :, t] = torchvision.transforms.functional.resize(video[:, :, t], resizing_shape)
    video_cropped = torch.zeros((B, C, T, target_h, target_w), dtype=video.dtype, device=video.device)
    for t in range(T):
        video_cropped[:, :, t] = torchvision.transforms.functional.center_crop(video_resized[:, :, t], resolution)
    return video_cropped

class AnticipativeWrapperNoAR(torch.nn.Module):
    """ Use predictor for inference """

    def __init__(
        self,
        pipeline,
        frames_per_clip=33,
        nb_context_frames=9, 
        args = None,   
        sampling_config=None,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.frames_per_clip = frames_per_clip
        self.nb_context_frames = nb_context_frames
        self.args = args
        self.sampling_config = sampling_config

    def forward(self, x):
        """
        :param x: (Tensor) video of shape [B, C, T, H, W]
        """
        logger.info(x.shape)
        # ==================================================
        #               Prepare videos for model
        # ==================================================
        device = x.device
        mean = torch.as_tensor((0.485, 0.456, 0.406)).to(device)[None, :, None, None, None]
        std = torch.as_tensor((0.229, 0.224, 0.225)).to(device)[None, :, None, None, None]
        unnorm_videos = x * std + mean  # in [0, 1]
        input_videos = unnorm_videos*2 -1


        logger.info(f"{input_videos.shape}")
        input_videos = resize_input(input_videos, self.args.data_resolution)
        #(batch_size, time, channels=3, height, width)
        input_videos = input_videos.permute(0,2,1,3,4)

        context_inputs = input_videos[:,:self.nb_context_frames]
        target_frames  = input_videos[:,self.nb_context_frames:]


        NUM_TOTAL_FRAMES = self.frames_per_clip
        num_input_frames = self.nb_context_frames
        context_inputs = torch.cat(
                (context_inputs, context_inputs[:,-1, :, :, :].unsqueeze(1).repeat(1,NUM_TOTAL_FRAMES - num_input_frames, 1, 1, 1)),
                dim=1,
            )
        context_inputs = context_inputs.permute(0,2,1,3,4)

        # ==================================================
        #               Compute targets
        # ==================================================


        latent_shape = self.pipeline.latent_shape
        input_videos = input_videos.permute(0,2,1,3,4)
        targets = []
        for i in range(len(input_videos)):
            data_batch = {"video": input_videos[i].unsqueeze(0)}
            data_batch = misc.to(data_batch, "cuda")
            data_tokens, token_boundaries = self.pipeline.model.tokenizer.tokenize(data_batch=data_batch)
            indices_tensor = data_tokens.long()
            codes = self.pipeline.model.tokenizer.video_tokenizer.tokenizer_module.quantizer.implicit_codebook[indices_tensor]
            log.info(f"Code shape before reshape: {codes.shape}")
            #Then we reshape to make slicing easier
            codes = rearrange(
            codes,
            "B (T H W) d -> B T H W d",
            T=latent_shape[0],
            H=latent_shape[1],
            W=latent_shape[2],
            )

            print(f"data_tokens: {codes.shape}")
            targets.append(torch.Tensor(codes[0,1:]).cpu())
        targets = torch.stack(targets)

        # ==================================================
        #               Compute predictions
        # ==================================================
        preds = []
        for i in range(len(context_inputs)):
            pred = self.pipeline.generate(
                inp_vid=context_inputs[i].unsqueeze(0),
                num_input_frames=self.args.num_input_frames,
                seed=self.args.seed,
                sampling_config=self.sampling_config,
            )
            # Due to the temporal compression factor of 8, things are quite messy.
            # Skipping one element is roughly equivalent to skipping the 9 context frames
            preds.append(torch.Tensor(pred[1:]).cpu())
        preds = torch.stack(preds)
       

        return preds.flatten(1,3).to(torch.float16), targets.flatten(1,3).to(torch.float16)
