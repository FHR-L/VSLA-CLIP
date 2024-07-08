from __future__ import print_function, absolute_import

import os
import random

import math
import torch
import functools
import torch.utils.data as data
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader

def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class VideoDataset(data.Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        img_paths, pid, camid = self.dataset[index]

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        # print(img_paths)
        random.shuffle(img_paths)
        # print("=================")
        # print(img_paths)
        # assert 1
        #
        clip = self.loader(img_paths)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        # print(clip)
        # assert 1 < 0


        return clip, pid, camid


class VideoDatasetInfer(data.Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 dataset,
                 seq_len=12,
                 temporal_sampler='restricted',
                 spatial_transform=None,
                 get_loader=get_default_video_loader):
        self.dataset = dataset
        self.seq_len = seq_len
        self.temporal_sampler = temporal_sampler
        self.spatial_transform = spatial_transform

        self.loader = get_loader()


    def __len__(self):
        return len(self.dataset)


    @staticmethod
    def loop_padding(img_paths, size):
        img_paths = list(img_paths)
        exp_len = math.ceil(len(img_paths) / size) * size
        while len(img_paths) != exp_len:
            lack_num = exp_len - len(img_paths)
            if len(img_paths) > lack_num:
                img_paths.extend(img_paths[-lack_num:])
            else:
                img_paths.extend(img_paths)

        img_paths.sort()
        assert len(img_paths) % size == 0, \
            'every clip must have {} frames, but we have {}' \
                .format(size, len(img_paths))
        return img_paths


    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        img_paths = self.loop_padding(img_paths, self.seq_len)

        clip = self.loader(img_paths)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        # C x T x H x W
        clip = torch.stack(clip, 1)
        C, T, H, W = clip.size()

        # T//seq_len, C, seq_len, H, W
        if self.temporal_sampler == 'restricted':
            clip = clip.reshape(C, self.seq_len, T // self.seq_len, H, W).permute(2, 0, 1, 3, 4)
        else:
            raise NotImplementedError

        return clip, pid, camid
