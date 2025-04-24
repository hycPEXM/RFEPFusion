from typing import Optional, Sequence, Union, List
from torch import Tensor
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from functools import partial

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss
# from loss_util import weighted_loss

from basicsr.losses.lovasz import lovasz_softmax

__all__ = ['ColorLoss', 'SSIMLoss', 'IntensityLoss', 'TextureLoss', 'SegLoss', 'HybridSegLoss']

_reduction_modes = ['none', 'mean', 'sum']

# TORCH_EPS = torch.finfo(torch.float32).eps

# 函参命名规则：ir, vi, fusion, mask_person, Y_vi, Y_fusion, seg_result, seg_label

@weighted_loss
def color_cosine_similarity_loss(pred, target):
    return 1 - F.cosine_similarity(pred, target, dim=1)  # 没有reduction参数
    # 两个(N, C, H, W)张量返回(N, H, W)

@LOSS_REGISTRY.register()
class ColorLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', mask=False, choice='vi', **kwargs):
        super(ColorLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.choice = choice
        self.mask = mask

    def forward(self, vi, fusion, enhanced, mask_person, weight=None, **kwargs):
        # print(fusion)        
        if self.mask:
            mask = ((mask_person-1).abs()<1e-6).repeat(1,3,1,1)
            fusion[mask] = 0
            enhanced[mask] = 0
            vi[mask] = 0
        if self.choice == 'vi':
            return self.loss_weight * color_cosine_similarity_loss(vi, fusion, weight, reduction=self.reduction)
        elif self.choice == 'enhanced':
            return self.loss_weight * color_cosine_similarity_loss(enhanced, fusion, weight, reduction=self.reduction)
        else:
            raise ValueError(f'Unsupported choice mode: {self.choice}. Supported ones are: [vi, enhanced]')

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)      # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window

# 方差计算
def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

# 计算 ssim 损失函数
def _ssim(img1, img2, window_size=11, max_val=255):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    # max_val = max_val
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2


    (_, channel, _, _) = img1.size()

    # 滤波器窗口
    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    # cs = torch.mean(v1 / v2)  # contrast sensitivity
    # ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    # ret = ssim_map
    # return ret
    return ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

def ssim_loss(img_ir, img_vis, img_fuse, weighted = True):

    ssim_ir = _ssim(img_ir, img_fuse, max_val=1)
    ssim_vi = _ssim(img_vis, img_fuse, max_val=1)

    if weighted:
        std_ir = std(img_ir)
        std_vi = std(img_vis)

        zero = torch.zeros_like(std_ir)
        one = torch.ones_like(std_vi)

        # m = torch.mean(img_ir)
        # w_ir = torch.where(img_ir > m, one, zero)

        map1 = torch.where((std_ir - std_vi) > 0, one, zero)
        # map2 = torch.where((std_ir - std_vi) >= 0, zero, one)
        map2 = 1 - map1
        map1 = map1.to(img_ir.device)
        map2 = map2.to(img_ir.device)

        ssim = 1 - (map1 * ssim_ir + map2 * ssim_vi)
        # ssim = ssim * w_ir
        # return ssim.mean()
    else:
        ssim = 1 - (ssim_ir + ssim_vi)/2
    return ssim.mean()


# Correlation Coefficient
class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()  
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r
   
    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)

# hyc认为应该逐图像的计算，这样img1.mean()是对一整个批次求平均
# 但似乎两种计算方式的结果差不多
def corr_loss(image_ir, img_vis, img_fusion, eps=1e-8):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss

def batch_corr_loss(ir, vi, fusion, eps=1e-8):
    reg = REG()
    b = ir.shape[0]
    corr = 0
    for i in range(b):
        corr += reg(ir[i], vi[i], fusion[i])
    corr /= b
    return 1./(corr + eps)


# only support 'mean' reduction
@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, choice='weighted', reduction='mean', **kwargs):
        super(SSIMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        # self.reduction = reduction
        if choice == 'weighted':
            self._loss_func = partial(ssim_loss, weighted=True)
        elif choice == 'unweighted':
            self._loss_func = partial(ssim_loss, weighted=False)
        elif choice == 'corr':
            self._loss_func = batch_corr_loss
        else:
            raise ValueError(f'Unsupported choice mode: {choice}. Supported ones are: [weighted, unweighted, corr]')
    def forward(self, ir, Y_vi, Y_fusion, weight=None, **kwargs):
        return self.loss_weight * self._loss_func(ir, Y_vi, Y_fusion)
    

@LOSS_REGISTRY.register()
class IntensityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', lambda_ir=1.5, 
                 lambda_vi=1.0, choice='Y', **kwargs):
        super(IntensityLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        # self.reduction = reduction
        self.lambda_ir = lambda_ir
        self.lambda_vi = lambda_vi
        self.choice = choice

    def forward(self, ir, enhanced, Y_enhanced, fusion, Y_fusion, mask_person, weight=None, **kwargs):
        # 让“人”在融合图像中的强度更接近于红外图像
        # weight_ir = (mask_person + self.lambda_ir * ir)/(1 + self.lambda_ir * (ir + Y_enhanced)+1e-8)
        if self.choice == 'Y':
            weight_ir = torch.where((mask_person-1).abs() < 1e-6, (mask_person + self.lambda_ir * ir)/(1 + self.lambda_ir * (ir + Y_enhanced) + 1e-8), ir/(ir + Y_enhanced + 1e-8))
            # 其他就按正常的权重计算
            weight_vi = self.lambda_vi * Y_enhanced/(ir + Y_enhanced + 1e-8)
            # weight_vi = 1 - weight_ir
            return self.loss_weight * ((weight_ir * (ir - Y_fusion)).abs().mean() + (weight_vi * (Y_enhanced - Y_fusion)).abs().mean())
        elif self.choice == 'RGB':
            weight_ir = torch.where((mask_person-1).abs() < 1e-6, (mask_person + self.lambda_ir * ir)/(1 + self.lambda_ir * (ir + Y_enhanced) + 1e-8), ir/(ir + Y_enhanced + 1e-8))
            weight_vi = self.lambda_vi * Y_enhanced/(ir + Y_enhanced + 1e-8)
            return self.loss_weight * ((weight_ir * (ir - Y_fusion)).abs().mean() + (weight_vi * (enhanced - fusion)).abs().mean())
        elif self.choice == 'Y_':
            weight_ir = (mask_person + self.lambda_ir * ir)/(1 + self.lambda_ir * (ir + Y_enhanced) + 1e-8)
            weight_vi = (1 - weight_ir) * self.lambda_vi
            # return self.loss_weight * ((weight_ir * (ir - Y_fusion)).abs().mean() + (weight_vi * (Y_enhanced - Y_fusion)).abs().mean())
            return self.loss_weight * (weight_ir * F.l1_loss(ir, Y_fusion) + 
                                       weight_vi * F.l1_loss(Y_enhanced, Y_fusion)).mean()
        elif self.choice == 'RGB_':
            weight_ir = (mask_person + self.lambda_ir * ir)/(1 + self.lambda_ir * (ir + Y_enhanced) + 1e-8)
            weight_vi = (1 - weight_ir) * self.lambda_vi
            return self.loss_weight * ((weight_ir * (ir - Y_fusion)).abs().mean() + (weight_vi * (enhanced - fusion)).abs().mean())
        else:
            raise ValueError(f'Unsupported choice mode: {self.choice}. Supported ones are: [Y, RGB, Y_, RGB_]')

@LOSS_REGISTRY.register()
class AuxiliaryIntensityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', 
                 choice='mask_based', **kwargs):
        super(AuxiliaryIntensityLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        # self.reduction = reduction
        self.choice = choice

    def forward(self, ir, Y_enhanced, Y_fusion, Y_vi, mask_person, weight=None, **kwargs):
        if self.choice == 'mask_based':
            std_vi = std(Y_vi) # 这里算的是vi而不是enhanced！
            std_ir = std(ir)
            zero = torch.zeros_like(std_ir)
            one = torch.ones_like(std_ir)
            map_ir = torch.where(std_ir>std_vi, one, zero)
            map_ir = torch.where(map_ir+mask_person>0, one, zero)
            map_vi = 1 - map_ir
            return self.loss_weight * (map_ir * F.l1_loss(Y_fusion, ir) + map_vi * F.l1_loss(Y_fusion, Y_enhanced)).mean()
        elif self.choice == 'max':
            return self.loss_weight * F.l1_loss(Y_fusion, torch.max(Y_enhanced, ir))            
        else:
            raise ValueError(f'Unsupported choice mode: {self.choice}. Supported ones are: [mask_based, max]')

    
class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        # 这里不行就采用expend_dims
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)

@LOSS_REGISTRY.register()
class TextureLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', device='cpu', **kwargs):
        super(TextureLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.device = device
        self.sobel = Sobelxy(self.device)
        # self.reduction = reduction
    def forward(self, ir, Y_vi, Y_fusion, weight=None, **kwargs):
        
        ir_grad_x, ir_grad_y = self.sobel(ir)
        vi_grad_x, vi_grad_y = self.sobel(Y_vi)
        fusion_grad_x, fusion_grad_y = self.sobel(Y_fusion)
        return self.loss_weight * (F.l1_loss(fusion_grad_x, torch.max(ir_grad_x, vi_grad_x)) 
                                   + F.l1_loss(fusion_grad_y, torch.max(ir_grad_y, vi_grad_y)))

#################################################
# Segmentation Loss
#################################################

# adapted from mmseg(SegNeXt)
# avg_factor is not used
# weight和reduction参数与weighted_loss里的wrapper参数重复了
@weighted_loss
def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # # apply weights and do the reduction
    # # average loss over non-ignored elements
    # # pytorch's official cross_entropy average loss over non-ignored elements
    # # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    # if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
    #     avg_factor = label.numel() - (label == ignore_index).sum().item()
    # if weight is not None:
    #     weight = weight.float()
    # loss = weight_reduce_loss(
    #     loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

# @LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', class_weight=None, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.class_weight = class_weight

    def forward(self, seg_result, seg_label, weight=None, **kwargs):
        return self.loss_weight * cross_entropy(seg_result, seg_label, weight, class_weight=self.class_weight, reduction=self.reduction)

# adapted from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
# @LOSS_REGISTRY.register()
class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Union[Tensor, List[float]]] = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100,
                 num_class: int = 9,
                 balance_index: int = 0,  # 对背景类施以较小的惩罚
                 **kwargs):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')
        
        if alpha is None:
           alpha_ = torch.ones(num_class, 1)
        elif isinstance(alpha, list):
            assert len(alpha) == num_class
            alpha_ = Tensor(alpha).view(num_class, 1)
            # 归一化之后使整个FocalLoss都除以了num_class
            # alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha_ = torch.ones(num_class, 1)
            alpha_ = alpha_ * (1 - alpha)
            alpha_[balance_index] = alpha
        elif isinstance(alpha, Tensor):
            if alpha.dim() == 1:
                alpha_ = alpha.view(-1, 1)
            elif alpha.dim() == 2:
                alpha_ = alpha
            else:
                raise ValueError('Invalid alpha shape')
        else:
            raise TypeError('Not supported data type for alpha')

        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha_, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl

# 另一种实现方式
# class AnotherFocalLoss(nn.Module):
#    """
#    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
#    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
#    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
#        Focal_Loss= -1*alpha*(1-pt)*log(pt)
#    :param num_class:
#    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
#    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
#                    focus on hard misclassified example
#    :param smooth: (float,double) smooth value when cross entropy
#    :param balance_index: (int) balance class index, should be specific when alpha is float
#    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
#    """

#    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
#        super(FocalLoss, self).__init__()
#        self.apply_nonlin = apply_nonlin
#        self.alpha = alpha
#        self.gamma = gamma
#        self.balance_index = balance_index
#        self.smooth = smooth
#        self.size_average = size_average

#        if self.smooth is not None:
#            if self.smooth < 0 or self.smooth > 1.0:
#                raise ValueError('smooth value should be in [0,1]')

#    def forward(self, logit, target):
#        if self.apply_nonlin is not None:
#            logit = self.apply_nonlin(logit)
#        num_class = logit.shape[1]

#        if logit.dim() > 2:
#            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
#            logit = logit.view(logit.size(0), logit.size(1), -1)
#            logit = logit.permute(0, 2, 1).contiguous()
#            logit = logit.view(-1, logit.size(-1))
#        target = torch.squeeze(target, 1)
#        target = target.view(-1, 1)
#        # print(logit.shape, target.shape)
#        # 
#        alpha = self.alpha

#        if alpha is None:
#            alpha = torch.ones(num_class, 1)
#        elif isinstance(alpha, (list, np.ndarray)):
#            assert len(alpha) == num_class
#            alpha = torch.FloatTensor(alpha).view(num_class, 1)
#            alpha = alpha / alpha.sum()
#        elif isinstance(alpha, float):
#            alpha = torch.ones(num_class, 1)
#            alpha = alpha * (1 - self.alpha)
#            alpha[self.balance_index] = self.alpha

#        else:
#            raise TypeError('Not support alpha type')
       
#        if alpha.device != logit.device:
#            alpha = alpha.to(logit.device)

#        idx = target.cpu().long()

#        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
#        one_hot_key = one_hot_key.scatter_(1, idx, 1)
#        if one_hot_key.device != logit.device:
#            one_hot_key = one_hot_key.to(logit.device)

#        if self.smooth:
#            one_hot_key = torch.clamp(
#                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
#        pt = (one_hot_key * logit).sum(1) + self.smooth
#        logpt = pt.log()

#        gamma = self.gamma

#        alpha = alpha[idx]
#        alpha = torch.squeeze(alpha)
#        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

#        if self.size_average:
#            loss = loss.mean()
#        else:
#            loss = loss.sum()
#        return loss

# soft jaccard loss
def jaccard_loss(logits, true, eps=1e-7, **kwargs):
    """Computes the Jaccard loss, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.

    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        # true_1_hot_f = true_1_hot[:, 0:1, :, :]
        # true_1_hot_s = true_1_hot[:, 1:2, :, :]
        # true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

# https://blog.csdn.net/weixin_44878336/article/details/132300255
class OhemCELoss(nn.Module):
    def __init__(self, thresh, ignore_lb=255, ignore_simple_sample_factor=16, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        # self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.thresh = -np.log(thresh)
        # self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.ignore_factor = ignore_simple_sample_factor
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels, **kwargs):
        # N, C, H, W = logits.size()
        n_min = labels[labels != self.ignore_lb].numel() // self.ignore_factor
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:n_min]
        return torch.mean(loss)
        # another implementation:
        # 选出loss大于阈值的像素点（困难样本）
        # loss_hard = loss[loss > self.thresh]
        # 如果总数小于n_min，至少要保留有n_min个像素点参与loss的计算
        # if loss_hard.numel() < n_min:
        #     loss_hard, _ = loss.topk(n_min)
        # return loss_hard.mean()        

@LOSS_REGISTRY.register()
class SegLoss(nn.Module):
    def __init__(self, loss_weight=1.0, choice='OHEM', num_class=9, reduction='mean', **kwargs):
        super(SegLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if choice == 'CE':
            # 可以统计一下每个类别出现的样本数和比例，然后设置class_weight
            self.loss = CrossEntropyLoss(reduction=reduction)
        elif choice == 'focal':
            background_index = 0
            self.loss = FocalLoss(alpha=0.25, gamma=2, balance_index=background_index, reduction=reduction, num_class=num_class, ignore_index=255)
        elif choice == 'OHEM':
            # 可以试一下ignore_simple_sample_factor=4或8
            # self.loss = OhemCELoss(thresh=kwargs['thresh'], ignore_lb=255, ignore_simple_sample_factor=kwargs['ignore_simple_sample_factor'])
            ignore_simple_sample_factor = kwargs.get('ignore_simple_sample_factor', 16)
            self.loss = OhemCELoss(thresh=0.75, ignore_lb=255, ignore_simple_sample_factor=ignore_simple_sample_factor)
        elif choice == 'lovasz':
            self.loss = lovasz_softmax
        elif choice == 'jaccard':
            self.loss = jaccard_loss

    def forward(self, seg_result, seg_label, weight=None, **kwargs):
        # return self.loss_weight * self.loss(seg_result, seg_label, weight, reduction=self.reduction)
        if seg_label.ndim == 4:
            seg_label = seg_label.squeeze(1)
        return self.loss_weight * self.loss(seg_result, seg_label)

@LOSS_REGISTRY.register()
class HybridSegLoss(nn.Module):
    def __init__(self, loss_weights : list =[1.0], choices=['OHEM'], num_class=9, reduction='mean', **kwargs):
        super(HybridSegLoss, self).__init__()
        assert len(choices) == len(loss_weights)
        self.loss_weights = loss_weights
        self.losses = []
        self.reduction = reduction
        for choice in choices:
            if choice == 'CE':
                self.losses.append(CrossEntropyLoss(reduction=reduction))
            elif choice == 'focal':
                background_index = 0
                self.losses.append(FocalLoss(alpha=0.25, gamma=2, balance_index=background_index, reduction=reduction, num_class=num_class))
            elif choice == 'OHEM':
                # 可以试一下ignore_simple_sample_factor=4或8
                self.losses.append(OhemCELoss(thresh=0.75, ignore_lb=255, ignore_simple_sample_factor=16))
            elif choice == 'lovasz':
                self.losses.append(lovasz_softmax)
            elif choice == 'jaccard':
                self.losses.append(jaccard_loss)

    def forward(self, seg_result, seg_label, weight=None, **kwargs):
        loss_total = 0
        # for i in range(len(self.losses)):
        #     loss_total += self.loss_weights[i] * self.losses[i](seg_result, seg_label, weight, reduction=self.reduction)
        for loss_weight, loss in zip(self.loss_weights, self.losses):
            loss_total += loss_weight * loss(seg_result, seg_label, weight, reduction=self.reduction)
        return loss_total

import kornia.filters as KF

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss,self).__init__()
        self.AP5 = nn.AvgPool2d(5,stride=1,padding=2).cuda()
        self.MP5 = nn.MaxPool2d(5,stride=1,padding=2).cuda()
    def forward(self,img1,img2,mask=1,eps=1e-2):
        #img1 = KF.gaussian_blur2d(img1,[7,7],[2,2])
        # mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
        mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
        mean_ = mean_.detach()/2
        std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
        std_ = std_.detach()/2 
        img1 = (img1-mean_)/std_
        img2 = (img2-mean_)/std_
        grad1 = KF.spatial_gradient(img1,order=2)
        grad2 = KF.spatial_gradient(img2,order=2)
        mask = mask.unsqueeze(1)
        # grad1 = self.AP5(self.MP5(grad1))
        # grad2 = self.AP5(self.MP5(grad2))
        # print((grad1-grad2).abs().mean())
        l = (((grad1-grad2)+(grad1-grad2).pow(2)*10)*mask).abs().clamp(min=eps).mean()
        #l = l[...,5:-5,10:-10].mean()
        return l

def l1loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,[3,3],(1,1))*mask_
    img2 = KF.gaussian_blur2d(img2,[3,3],(1,1))*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).mean()

def l2loss(img1,img2,mask=1,eps=1e-2):
    mask_ = torch.logical_and(img1>1e-2,img2>1e-2)
    mean_ = img1.mean(dim=[-1,-2],keepdim=True)+img2.mean(dim=[-1,-2],keepdim=True)
    mean_ = mean_.detach()/2
    std_ = img1.std(dim=[-1,-2],keepdim=True)+img2.std(dim=[-1,-2],keepdim=True)
    std_ = std_.detach()/2 
    img1 = (img1-mean_)/std_
    img2 = (img2-mean_)/std_
    img1 = KF.gaussian_blur2d(img1,[3,3],(1,1))*mask_
    img2 = KF.gaussian_blur2d(img2,[3,3],(1,1))*mask_
    return ((img1-img2)*mask).abs().clamp(min=eps).pow(2).mean()

@LOSS_REGISTRY.register()
class PhotometricLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super().__init__()
        self.grad_loss = GradientLoss()
        self.loss_weight = loss_weight
    def forward(self, src, tgt, mask=1, weights=[0.1, 0.9], **kwargs):
        return self.loss_weight * (weights[0] * (l1loss(src, tgt, mask) + l2loss(src, tgt, mask)) + weights[1] * self.grad_loss(src, tgt, mask))
        
@LOSS_REGISTRY.register()
class EndPointLoss(nn.Module):
    def __init__(self, loss_weight=1.0, dynamic_shape = False, *args, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.dynamic_shape = dynamic_shape
    def forward(self, ref, tgt, disp, disp_gt, **kwargs):
        if self.dynamic_shape:
            _, _, h, w = ref.shape
            self.border_mask = torch.zeros([1,h,w,1]).to(ref.device)
            self.border_mask[:, 10:-10, 10:-10, :] = 1
        elif not hasattr(self, 'border_mask'):
            _, _, h, w = ref.shape
            self.border_mask = torch.zeros([1,h,w,1]).to(ref.device)
            self.border_mask[:, 10:-10, 10:-10, :] = 1
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(-1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(-1)
        # print(g_ref.shape)
        w = (((g_ref + g_tgt)) * 2 + 1) * self.border_mask
        return self.loss_weight * (w * (1000 * (disp - disp_gt).abs().clamp(min=1e-3).pow(2))).mean()

# re指代regularization
def smooth_loss(disp,img=None):
    smooth_d=[3*3,7*3,15*3]
    # b,c,h,w = disp.shape
    # print(disp.shape)
    grad = KF.spatial_gradient(disp,order=2).abs().sum(dim=2)[:,:,5:-5,5:-5].clamp(min=1e-9).mean()
    local_smooth_re = 0
    for d in smooth_d:
        local_mean = KF.gaussian_blur2d(disp,[d,d],(d//6,d//6),border_type='replicate')
        #local_mean_pow2 = F.avg_pool2d(disp.pow(2),kernel_size=d,stride=1,padding=d//2)
        local_smooth_re += 1/(d*1.0+1)*(disp-local_mean)[:,:,d//2:-d//2,d//2:-d//2].pow(2).mean()
        #local_smooth_re += 1/(d*1.0+1)*(disp.pow(2)-local_mean_pow2)[:,:,5:-5,5:-5].pow(2).mean()
    #global_var = disp[...,2:-2,2:-2].var(dim=[-1,-2]).clamp(1e-5).mean()
    #std = img.std(dim=[-1,-2]).mean().clamp(min=0.003)
    #grad = grad[...,10:-10,10:-10]
    return 5000*local_smooth_re + 500*grad

@LOSS_REGISTRY.register()
class DefRegLoss(nn.Module):
    def __init__(self, loss_weight=1.0, *args, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, disp):
        return self.loss_weight * smooth_loss(disp)
    
def border_suppression(img, mask, **kwargs):
        return (img * (1 - mask)).mean()
    
@LOSS_REGISTRY.register()
class BorderSuppressionLoss(nn.Module):
    def __init__(self, loss_weight=1.0, *args, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
    def forward(self, img, mask):
        return self.loss_weight * (img * (1 - mask)).mean()
        
if __name__ == '__main__':
    ir = torch.rand([5, 1, 320, 240])
    Y_vi = torch.rand([5, 1, 320, 240])
    Y_fusion = torch.rand([5, 1, 320, 240])
    print(corr_loss(ir, Y_vi, Y_fusion))
    print(batch_corr_loss(ir, Y_vi, Y_fusion))
    torch.set_printoptions(precision=6)
    weighted_ssim = ssim_loss(ir, Y_vi, Y_fusion)
    print(weighted_ssim.shape, weighted_ssim.mean())
    unweighted_ssim = ssim_loss(ir, Y_vi, Y_fusion, weighted=False)
    print(unweighted_ssim.shape, unweighted_ssim.mean())