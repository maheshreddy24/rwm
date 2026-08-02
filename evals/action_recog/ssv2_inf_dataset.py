import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from decord import VideoReader, cpu, gpu
from PIL import Image
import yaml
import json

class SSv2(Dataset):
    def __init__(self, config, split='train'):
        super().__init__()

        self.split = split
        data_json_path = config[f'{split}_json']
        with open(data_json_path) as f:
            self.data = json.load(f)
        with open(config['labels_json']) as f:
            self.labels = json.load(f)

        self.video_paths = []
        self.class_idx = []

        for item in self.data:
            video_path = f"{config['videos_dir']}/{item['video_id']}.mp4"
            self.video_paths.append(video_path)
            self.class_idx.append(self.labels[item['template'].replace("[", "").replace("]", "")])

        print(f"Loaded {len(self.video_paths)} videos for {split} split.")

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(239),
                transforms.RandomCrop(224),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.1),
            ])
        else:
            self.transform = None


    def _load_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        frames = [vr[i].asnumpy() for i in range(len(vr))]
        if self.transform is not None:
            frames = [self.transform(Image.fromarray(frame)) for frame in frames]
        # return frames
        frames = frames[:self.config['num_frames'] * self.config['frame_stride']]
        frames = frames[::self.config['frame_stride']]
        return torch.stack(frames)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        class_idx = self.class_idx[idx]
        frames = self._load_video(video_path)
        return frames, class_idx

    def __len__(self):
        return len(self.video_paths)
    
        