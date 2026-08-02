import json
import yaml
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from decord import VideoReader, cpu
from PIL import Image


class SSv2(Dataset):
    def __init__(self, config_path, split="train"):
        super().__init__()

        self.config = yaml.safe_load(open(config_path, "r"))
        self.split = split

        with open(self.config[f"{split}_json"], "r") as f:
            self.data = json.load(f)

        with open(self.config["labels_json"], "r") as f:
            self.labels = json.load(f)

        self.video_paths = []
        self.class_idx = []

        for item in self.data:
            self.video_paths.append(
                f"{self.config['videos_dir']}/{item['video_id']}.mp4"
            )

            label = item["template"].replace("[", "").replace("]", "")
            self.class_idx.append(self.labels[label])

        print(f"Loaded {len(self.video_paths)} videos for {split} split.")

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize(239),
                transforms.RandomCrop(224),
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    )],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.1),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(239),
                transforms.CenterCrop(224),
            ])

    def _load_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))

        total_frames = len(vr)
        num_frames = self.config["num_frames"]
        stride = self.config["frame_stride"]

        required = num_frames * stride

        if total_frames >= required:
            indices = list(range(0, required, stride))
        else:
            # Take as many frames as possible with the given stride
            indices = list(range(0, total_frames, stride))

        frames = []

        for idx in indices:
            img = Image.fromarray(vr[idx].asnumpy())
            img = self.transform(img)

            # HWC float tensor in [0, 1]
            img = torch.from_numpy(np.array(img)).float() / 255.0
            frames.append(img)

        # (T, H, W, C)
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        frames = self._load_video(self.video_paths[idx])
        label = self.class_idx[idx]
        return frames, label

    def __len__(self):
        return len(self.video_paths)