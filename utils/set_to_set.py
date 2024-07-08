import torch
from torch.autograd.grad_mode import F


def similarity_calculation(feat1: torch.Tensor, feat2: torch.Tensor):
    feat1 = F.normalize(feat1, dim=1, p=2)
    feat2 = F.normalize(feat2, dim=1, p=2)
    distmat = -torch.matmul(feat1, feat2.transpose(0, 1))
    print(distmat)
