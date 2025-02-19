import torch
from torch import nn as nn
from torch.nn import functional as F


from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def color_cosine_similarity_loss(pred, target):
    return 1 - F.cosine_similarity(pred, target, dim=1)

@LOSS_REGISTRY.register()
class ColorLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ColorLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * color_cosine_similarity_loss(pred, target, weight, reduction=self.reduction)

