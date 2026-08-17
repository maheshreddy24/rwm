# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from PIL import Image
import torch
import json
from einops import rearrange
from decord import VideoReader,cpu
from pathlib import Path


_GLOBAL_SEED = 0
logger = getLogger()


def make_videodataset(
    data_path,
    batch_size,
    frame_step=4,
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    deterministic=True,
    log_dir=None,
):    
    dataset = IntPhys2Dataset(
    data_path=data_path,
    frame_step=frame_step,
    transform=transform)

    log_dir = pathlib.Path(log_dir) if log_dir else None

    logger.info('Dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=drop_last)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=True)
    logger.info('Intuitive Physics data loader created')

    return dataset, data_loader, dist_sampler


class IntPhys2Dataset(torch.utils.data.Dataset):
    """ IntPhys2 dataset """

    def __init__(
        self,
        data_path,
        frame_step=10,
        transform=None,
        use_image_dir=False
    ):
    
        self.data_path = data_path
        labels = pd.read_csv(self.data_path + "/metadata.csv")
        videos = sorted(list(labels['file_name'].unique()))
        if use_image_dir:
            self.videopaths =  sorted([self.data_path +'/Images/'+ vid for vid  in videos if os.path.exists(self.data_path +'/Images/'+ vid )])
        else:
            self.videopaths =  sorted([self.data_path +'/'+ vid for vid  in videos if os.path.exists(self.data_path +'/'+ vid )])

        self.frame_step = frame_step
        self.transform = transform 
        self.use_image_dir = use_image_dir

    def __getitem__(self, index):

        if self.use_image_dir:
            frames_all = sorted(os.listdir(self.videopaths[index]))
            frames_to_load = frames_all[::self.frame_step]
    
            frames = []
            for frame in frames_to_load:
                frame_ = Image.open(f"{self.videopaths[index]}/{frame}")
                frame_ = torch.Tensor(np.array(frame_))
                if frame_.size(-1) == 4:
                    frame_ = frame_[...,:-1]
                frames.append(frame_)
            frames = torch.stack(frames)
        else:
            vr = VideoReader(self.videopaths[index], ctx=cpu(0))
            frame_indices = np.arange(0, len(vr), self.frame_step)
            frames = vr.get_batch(frame_indices).asnumpy()
            frames = torch.from_numpy(frames)

        if self.transform:
            frames = self.transform(frames)
        else:
            frames = torch.Tensor(frames)
        
        name =  torch.Tensor([index])
        return frames,name

    def __len__(self):
        return len(self.videopaths)
