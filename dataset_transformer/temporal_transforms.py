from __future__ import absolute_import

import random
import numpy as np


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = list(frame_indices)

        while len(out) < self.size:
            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)

        return out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, seq_len=4, sample_stride=8, **kwargs):
        self.size = seq_len
        self.stride = sample_stride

    def __call__(self, frame_indices, stride=None):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)
        self.stride = stride if stride is not None else self.stride

        if len(frame_indices) >= self.size * self.stride:
            rand_end = len(frame_indices) - (self.size - 1) * self.stride - 1
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + (self.size - 1) * self.stride + 1
            out = frame_indices[begin_index:end_index:self.stride]
        elif len(frame_indices) >= self.size:
            index = np.random.choice(len(frame_indices), size=self.size, replace=False)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]
        else:
            index = np.random.choice(len(frame_indices), size=self.size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]
        return out


class TemporalBeginCrop(object):

    def __init__(self, size=4, sample_stride=8, **kwargs):
        self.size = size
        self.stride = sample_stride

    def __call__(self, frame_indices):
        frame_indices = list(frame_indices)
        size = self.size
        stride = self.stride
        if len(frame_indices) >= size * stride:
            out = frame_indices[0:(size-1)*stride + 1: stride]

        elif len(frame_indices) >= size:
            out = frame_indices[:size]
        else:
            index = np.random.choice(len(frame_indices), size=size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(size)]
        return out


class TemporalRestrictedCrop(object):

    def __init__(self, size=4, **kwargs):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)

        while len(frame_indices) < self.size:
            frame_indices.append(frame_indices[-1])

        out = []
        block_size = len(frame_indices)//self.size
        for i in range(self.size - 1):
            index = i*block_size + random.randint(0, block_size-1)
            out.append(frame_indices[index])

        index = (self.size-1)*block_size + random.randint(0, len(frame_indices)-(self.size-1)*block_size-1)
        out.append(frame_indices[index])

        return out


class TemporalRestrictedBeginCrop(object):

    def __init__(self, size=4):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)

        while len(frame_indices) < self.size:
            frame_indices.append(frame_indices[-1])

        out = []
        block_size = len(frame_indices)//self.size
        for i in range(self.size):
            index = i*block_size
            out.append(frame_indices[index])
        out.sort()

        return out


tem_factory = {
    'random': TemporalRandomCrop,
    'begin': TemporalBeginCrop,
    'restricted': TemporalRestrictedCrop,
    'restrictedbegin': TemporalRestrictedBeginCrop,
}


