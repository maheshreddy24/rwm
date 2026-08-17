# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass
from pathlib import Path
import logging
import pprint

import numpy as np
from einops import rearrange
import importlib

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from datetime import timedelta


from app.vjepa.transforms import make_transforms

from evals.intphys2.utils import get_dataset_paths,batch_all_gather,pad_tensors
from evals.intphys2.data_manager import init_data

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    # -- DATA
    args_exp = args_eval.get("experiment")
    args_data = args_exp.get('data')
    resolution = args_data.get('resolution', 224)
    batch_size = args_data.get('batch_size', 1)
    stride_sliding_window = args_data.get('stride_sliding_window',2)
    use_bfloat16 = args_data.get('use_bfloat16')
    frames_per_clip = args_data.get('frames_per_clip', 16)

    all_context_lengths = args_data.get('context_lengths', 4)
    num_frames_to_pred = args_data.get('num_frames_to_pred',-1)
    frame_steps = args_data.get('frame_steps', 4)
    dataset = args_data.get('dataset', 'intphys2-main')

    # -- EXPERIMENT
    mode = args_exp.get('mode', 'all')
    assert mode in ['all','losses','metrics']
    max_context_mode = args_exp.get('max_context_mode',True)

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'intphys_v2/')
    if eval_tag is not None:
        folder = os.path.join(folder, f"{dataset}-{eval_tag}")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'log_r{rank}.csv')
    # Initialize model

    # -- pretrained encoder (frozen)
    model = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        nb_context_frames=1,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )

    # Initialize data loaders
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1/1, 1/1],
        random_resize_scale=[1.0, 1.0],
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=resolution)

    if not isinstance(frame_steps, list):
        frame_steps = [frame_steps]

    for frame_step in frame_steps:
        logger.info(f"Extracting losses ...")
        all_losses,all_ids,names  = extract_losses(
            device=device,
            model=model,
            transform=transform,
            use_bfloat16=use_bfloat16,
            frame_step=frame_step,
            context_lengths=all_context_lengths,
            batch_size=batch_size,
            frames_per_clip=frames_per_clip,
            stride=stride_sliding_window,
            world_size=world_size,
            rank=rank,
            dataset=dataset,
            resolution=resolution,
            num_frames_to_pred=num_frames_to_pred,
            max_context_mode=max_context_mode)
            
        if dataset != "intphys2-main":
            logger.info("Gathering everything")
            all_losses = batch_all_gather(all_losses).cpu()
            all_ids = batch_all_gather(all_ids).cpu().to(torch.int)

            all_names = np.array(names)[all_ids.tolist()]
        else:
            all_losses = all_losses.cpu()
            all_ids = all_ids.cpu().to(torch.int)

            all_names = np.array(names)[all_ids.tolist()]

        logger.info("Saving...")

        if dataset == "intphys2-main":
             torch.save({ "frame_step":frame_step,
                        "num_frames_to_pred":num_frames_to_pred,
                        "context_lengths":all_context_lengths,
                        "losses":all_losses,
                        "names":all_names,
                        },
                        os.path.join(folder, f'losses_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt_rank{rank}.pth'))
        elif rank == 0 :
            torch.save({ "frame_step":frame_step,
                        "num_frames_to_pred":num_frames_to_pred,
                        "context_lengths":all_context_lengths,
                        "losses":all_losses,
                        "names":all_names,
                        },
                        os.path.join(folder, f'losses_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt.pth'))            
    

@torch.no_grad()
def extract_losses(
    device,
    model,
    transform,
    use_bfloat16=False,
    frame_step=1,
    context_lengths=[2],
    batch_size=1,
    frames_per_clip=16,
    stride=2,
    world_size=1,
    rank=0,
    dataset="intphys2-main",
    resolution=224,
    num_frames_to_pred=-1,
    max_context_mode=False
):
    print(context_lengths)

    if dataset == "intphys2-debug":
        data_name = f"IntPhys2-debug"
    elif dataset == "intphys2-main":
        data_name = f"IntPhys2-main"
    elif dataset == "intphys2-heldout":
        data_name = f"IntPhys2-heldout"


    (data,unsupervised_sampler) = init_data(
        batch_size = batch_size,
        transform=transform,
        collator=None,
        pin_mem=True,
        num_workers=8,
        world_size=world_size,
        rank=rank,
        root_path=get_dataset_paths([data_name])[0],
        frame_sample_rate=frame_step,
        deterministic=True,
        log_dir=None,)


    loader = iter(data)

    all_ids = []
    all_losses = []

    for i in range(len(loader)):
        udata_labels = next(loader)

        id = udata_labels[1][0]

        clip = udata_labels[0]
        clip = clip.to(device)
        
        all_losses_ctxt = []
        for CTXT_LEN in context_lengths:
            logger.info("="*40)
            model.nb_context_frames=CTXT_LEN
            model.frames_per_clip = CTXT_LEN + num_frames_to_pred if num_frames_to_pred != -1 else frames_per_clip
            try:
                model.grid_depth = model.frames_per_clip // model.encoder.tubelet_size
            except:
                try:
                    model.grid_depth = model.frames_per_clip // model.encoder.backbone.tubelet_size
                except:
                    model.grid_depth = model.frames_per_clip // 2

            logger.info(f"CTXT {CTXT_LEN}, FPC = {model.frames_per_clip}")
            pieces = clip.unfold(2, model.frames_per_clip,stride).permute(0,2,-1,1,3,4).contiguous()

            pieces = pieces.flatten(0,1)#.view(-1,3,16,224,224)
            pieces = rearrange(pieces,"b t c h w ->  b c t h w")
            logger.info(f"pieces {pieces.shape}")

            pieces = pieces.contiguous()

            B, C, T, H, W = pieces.shape
            
            chunked_preds = []
            chunked_targets = []
            CHUNK_SIZE = 12

            if max_context_mode:
                losses_beginning = []
                logger.info("Doing smalle2 contexts")
                #Predict for all context smaller than CONTEXT, in case the physics breaking event happens before CONTEXT +1
                #Harcoded tubelet size of 2
                for ctxt in [2*i for i in range(1,CTXT_LEN//2)]:
                    #Update wrapper
                    model.nb_context_frames=ctxt
                    model.frames_per_clip = ctxt + num_frames_to_pred if num_frames_to_pred != -1 else frames_per_clip
                    try:
                        model.grid_depth = model.frames_per_clip // model.encoder.tubelet_size
                    except:
                        try:
                            model.grid_depth = model.frames_per_clip // model.encoder.backbone.tubelet_size
                        except:
                            model.grid_depth = model.frames_per_clip // 2

                    clip_beginning = clip[:,:,:model.frames_per_clip]
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                        chunk = clip_beginning
                        logger.info(f"ctxt: {ctxt}, chunk shape{chunk.shape}")

                        preds, targets = model(chunk)
                        l = F.l1_loss(preds,targets,reduction="none").mean((1,2)).detach().to(device)
                        losses_beginning.append(l)

                loss_beginning = torch.stack(losses_beginning)
                loss_beginning = rearrange(loss_beginning,"nvid ctxt -> ctxt nvid")
                logger.info("Finished doing smaller contexts")
                

            #Update wrapper
            model.nb_context_frames=CTXT_LEN
            model.frames_per_clip = CTXT_LEN + num_frames_to_pred if num_frames_to_pred != -1 else frames_per_clip
            logger.info(f"{num_frames_to_pred} frames to pred, {model.frames_per_clip} total frames ")
            try:
                model.grid_depth = model.frames_per_clip // model.encoder.tubelet_size
            except:
                try:
                    model.grid_depth = model.frames_per_clip // model.encoder.backbone.tubelet_size
                except:
                    model.grid_depth = model.frames_per_clip // 2

            # First prediction here is at CONTEXT +1
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                logger.info(f"Number of chunks {int(np.ceil(pieces.shape[0]/CHUNK_SIZE))}")
                for chunk_id in range(int(np.ceil(pieces.shape[0]/CHUNK_SIZE))):
                    chunk = pieces[CHUNK_SIZE*chunk_id:CHUNK_SIZE*(chunk_id+1)]

                    preds, targets = model(chunk)
                    chunked_preds.append(preds.cpu())
                    chunked_targets.append(targets.cpu())
                    
                preds = torch.vstack(chunked_preds)
                targets = torch.vstack(chunked_targets)
                preds = preds.unsqueeze(0)
                targets = targets.unsqueeze(0)

            loss = F.l1_loss(preds,targets,reduction="none").mean((2,3)).detach().to(device)
            logger.info(f"Loss: {loss.shape}")
            if max_context_mode:
                loss = torch.hstack([loss_beginning,loss])
            all_losses_ctxt.append(loss)

        for l in all_losses_ctxt:
            logger.info(l.shape)
        max_length = max([l.size(1) for l in all_losses_ctxt])
        logger.info(max_length)
        all_losses_ctxt = pad_tensors(all_losses_ctxt,max_length,length_axis=1)
        losses = torch.stack(all_losses_ctxt)
        losses = losses.permute(1,0,2)

        all_losses.append(losses)
        all_ids.append(id)

    logger.info("Losses computed, moving on to synchronisation")

    lengths = []
    for l in all_losses:
        lengths.append(l.size(2))
    max_length = torch.tensor([max(lengths)]).to(device)
    #We need to sync the max lengths otherwise we can't gather the losses afterwards
    if data_name != f"IntPhys2-main":
        dist.all_reduce(max_length, op=dist.ReduceOp.MAX)
    for l in all_losses:
        logger.info(l.shape)
    logger.info("Padding complete")
    logger.info(lengths)

    all_losses = pad_tensors(all_losses,max_length.item(),length_axis=2)
    for l in all_losses:
        logger.info(l.shape)

    all_losses = torch.concat(all_losses)
    all_ids = torch.concat(all_ids)


    return all_losses,all_ids.to(device),[vid.split('/')[-1] for vid in data.dataset.videopaths]


def init_module(
    module_name,
    device,
    frames_per_clip,
    nb_context_frames,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
):
    """
    Build (frozen) model and initialize from pretrained checkpoint

    API requirements for "model" module:
      1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :param anticipation_time: (Tensor) Seconds into the future to predict for each sample in batch (shape=[batch_size])
        :returns: (Tensor) Representations of future frames (shape=[batch_size x num_output_tokens x feature_dim])

      2) Needs to have a public attribute called 'embed_dim' (int) describing its
         output feature dimension.
    """
    model = importlib.import_module(f"{module_name}").init_module(
        frames_per_clip=frames_per_clip,
        nb_context_frames=nb_context_frames,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(model)
    return model



def init_distributed(port=37129, rank_and_world_size=(None, None)):
    # try to set all environment variables to avoid triggering a segfault
    # environment variables can be reallocated during the execution of torch.distributed.init_process_group
    # the idea is a race condition may trigger if init_progress_group is modifying an environment variable at
    # the same time as Python, so we try to set all environs before initializing distributed
    if "SLURM_JOB_ID" in os.environ:
        # Use the slurm_tmpdir (if it exists) instead of /tmp
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ["MASTER_ADDR"] = "localhost"

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["SLURM_PROCID"])
            os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        except Exception:
            logger.info("SLURM vars not set (distributed training not available)")
            world_size, rank = 1, 0
            return world_size, rank

    try:
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank,timeout=timedelta(seconds=3600))
    except Exception as e:
        world_size, rank = 1, 0
        logger.info(f"Rank: {rank}. Distributed training not available {e}")

    return world_size, rank
