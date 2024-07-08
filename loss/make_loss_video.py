import torch.nn.functional as F
from torch import nn

from loss.center_loss import CenterLoss
from loss.losses_video import CrossEntropyLabelSmooth, TripletLoss


def make_loss(cfg, num_classes):
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target, target_cam, i2tscore=None, loss_t2i=None, loss_i2t=None):
        criterions = {
            'xent': nn.CrossEntropyLoss(),
            'htri': TripletLoss(margin=0.3, distance=cfg.TEST.DISTANCE)}
        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score[0:]]
                    ID_LOSS = sum(ID_LOSS)
                else:
                    ID_LOSS = xent(score, target)

                if isinstance(feat, list):
                    TRI_LOSS = [criterions['htri'](feats, target) for feats in feat[0:]]
                    TRI_LOSS = sum(TRI_LOSS)
                else:
                    TRI_LOSS =criterions['htri'](feat, target)

                loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                if i2tscore != None and cfg.MODEL.I2T_LOSS_WEIGHT != 0:
                    I2TLOSS = xent(i2tscore, target)
                    loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                loss = loss + cfg.MODEL.I2T_WEIGHT * loss_i2t + cfg.MODEL.T2I_WEIGHT * loss_t2i

                return loss
            else:
                if isinstance(score, list):
                    ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                    ID_LOSS = sum(ID_LOSS)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

                if isinstance(feat, list):
                    TRI_LOSS = [criterions['htri'](feats, target)[0] for feats in feat[0:]]
                    TRI_LOSS = sum(TRI_LOSS)
                else:
                    TRI_LOSS = criterions['htri'](feat, target)[0]

                loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                if i2tscore != None and cfg.MODEL.I2T_LOSS_WEIGHT != 0:
                    I2TLOSS = F.cross_entropy(i2tscore, target)
                    loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                loss = loss + cfg.MODEL.I2T_WEIGHT * loss_i2t + cfg.MODEL.T2I_WEIGHT * loss_t2i
                return loss
    return loss_func, center_criterion