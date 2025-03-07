import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss
# from loss_util import weighted_loss

__all__ = ['ColorLoss', 'SSIMLoss', 'IntensityLoss', 'TextureLoss']

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def color_cosine_similarity_loss(pred, target):
    return 1 - F.cosine_similarity(pred, target, dim=1)  # 没有reduction参数
    # 两个(N, C, H, W)张量返回(N, H, W)

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

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
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

    max_val = max_val
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2


    (_, channel, height, width) = img1.size()

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

    ssim_ir = _ssim(img_ir, img_fuse)
    ssim_vi = _ssim(img_vis, img_fuse)

    if weighted:
        std_ir = std(img_ir)
        std_vi = std(img_vis)

        zero = torch.zeros_like(std_ir)
        one = torch.ones_like(std_vi)

        # m = torch.mean(img_ir)
        # w_ir = torch.where(img_ir > m, one, zero)

        map1 = torch.where((std_ir - std_vi) > 0, one, zero)
        map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

        ssim = 1 - (map1 * ssim_ir + map2 * ssim_vi)
        # ssim = ssim * w_ir
        # return ssim.mean()
    else:
        ssim = 1 - (ssim_ir + ssim_vi)/2
    return ssim


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


@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    
    
@LOSS_REGISTRY.register()
class IntensityLoss(nn.Module):


@LOSS_REGISTRY.register()
class TextureLoss(nn.Module):


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