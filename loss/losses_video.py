from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, distance='cosine'):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, input, target):
        """
        :param input: feature matrix with shape (batch_size, feat_dim)
        :param target:  ground truth labels with shape (batch_size)
        :return:
        """
        n = input.size(0)
        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'cosine':
            input = F.normalize(input, dim=-1)
            dist = - torch.matmul(input, input.t())
        else:
            raise NotImplementedError

        # For each anchor, find the hardest positive and negative
        mask = target.expand(n, n).eq(target.expand(n, n).t()).float()
        dist_ap, _ = torch.topk(dist*mask - (1-mask), dim=-1, k=1)
        dist_an, _ = torch.topk(dist*(1-mask) + mask, dim=-1, k=1, largest=False)


        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class InfoNce(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self,
                 temperature=0.07,
                 num_instance=4):

        super(InfoNce, self).__init__()
        self.temperature = temperature
        self.ni = num_instance

    def forward(self, features):
        """
        :param features: (B, C, T)
        :param labels: (B)
        :return:
        """
        b, c, t = features.shape
        if t == 8:
            features = features.reshape(b, c, 2, 4).transpose(1, 2).reshape(b*2, c, 4)
            b, c, t = features.shape

        ni = self.ni
        features = features.reshape(b//ni, ni, c, t).permute(0, 3, 1, 2).reshape(b//ni, t*ni, c)
        features = F.normalize(features, dim=-1)
        labels = torch.arange(0, t).reshape(t, 1).repeat(1, ni).reshape(t*ni, 1)
        # (t*ni, t*ni)
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().cuda()  # (t*ni, t*ni)
        mask_pos = (1 - torch.eye(t*ni)).cuda()
        mask_pos = (mask * mask_pos).unsqueeze(0)

        # (b//ni, t*ni, t*ni)
        cos = torch.matmul(features, features.transpose(-1, -2))

        logits = torch.div(cos, self.temperature)
        exp_neg_logits = (logits.exp() * (1-mask)).sum(dim=-1, keepdim=True)

        log_prob = logits - torch.log(exp_neg_logits + logits.exp())
        loss = (log_prob * mask_pos).sum() / (mask_pos.sum())
        loss = - loss
        return loss


if __name__ == '__main__':
    loss = InfoNce()
    x = torch.rand(8, 16, 4).cuda()
    y = loss(x)
    print(y)
