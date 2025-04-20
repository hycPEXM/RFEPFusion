import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from basicsr.utils.registry import METRIC_REGISTRY
import math

# https://github.com/hanna-xu/MURF/blob/main/RGB-IR/fine_registration_and_fusion/utils.py
# MURF是很值得参考的配准范例，还提供了grid_sample的从头实现，只可惜
# def defor_reverse(defor, half_window=10):
#     batchsize, height, width, _ = defor.shape
#     # x = np.linspace(-1.0, 1.0, height)
#     # y = np.linspace(-1.0, 1.0, width)
#     x = np.linspace(-1.0, 1.0, width)
#     y = np.linspace(-1.0, 1.0, height)
#     xx, yy = np.meshgrid(y, x)
#     xx = np.transpose(xx)
#     yy = np.transpose(yy)
#     xx = np.expand_dims(xx, -1)
#     yy = np.expand_dims(yy, -1)
#     xx = np.expand_dims(xx, 0)
#     yy = np.expand_dims(yy, 0)
#     identity = np.concatenate([yy, xx], axis=-1)
#     # print(identity.shape)
#     # print(defor.shape)
#     defor = defor.detach().cpu().numpy()
#     resampling_grid = identity + defor

#     defor_re = np.zeros_like(defor)

#     for h_index in range(height):
#         for w_index in range(width):
#             y_min = cmin(h_index - half_window)
#             y_max = cmax(h_index + half_window, height)
#             x_min = cmin(w_index - half_window)
#             x_max = cmax(w_index + half_window, width)
#             diff = np.square(resampling_grid[0:batchsize, y_min:y_max, x_min:x_max, 0] - np.tile(identity[0:batchsize, h_index, w_index, 0],
#                     (1, y_max-y_min, x_max-x_min))) + np.square(resampling_grid[0:batchsize, y_min:y_max, x_min:x_max, 1] -
#                     np.tile(identity[0:batchsize, h_index, w_index, 1], (1, y_max-y_min, x_max-x_min)))
#             for b in range(batchsize):
#                 index = diff[b, :, :].argmin()
#                 re_h = index // (x_max-x_min) + y_min
#                 re_w = index - index // (x_max-x_min) * (x_max - x_min) +x_min
#                 defor_re[b, h_index, w_index, 0] = - defor[b, re_h, re_w, 0]
#                 defor_re[b, h_index, w_index, 1] = - defor[b, re_h, re_w, 1]
#     return defor_re

def cmin(value):
    return max(0, value)

def cmax(value, max_value):
    return min(value, max_value)

def defor_reverse(defor, half_window=10):
    batchsize, height, width, _ = defor.shape
    # 修正 x 和 y 的赋值
    x = np.linspace(-1.0, 1.0, width)
    y = np.linspace(-1.0, 1.0, height)
    # 修正 meshgrid 的调用顺序
    xx, yy = np.meshgrid(x, y)
    # 去掉不必要的转置操作
    xx = np.expand_dims(xx, -1)
    yy = np.expand_dims(yy, -1)
    xx = np.expand_dims(xx, 0)
    yy = np.expand_dims(yy, 0)
    identity = np.concatenate([xx, yy], axis=-1)
    defor = defor.detach().cpu().numpy()
    resampling_grid = identity + defor

    defor_re = np.zeros_like(defor)

    for h_index in range(height):
        for w_index in range(width):
            y_min = cmin(h_index - half_window)
            y_max = cmax(h_index + half_window, height)
            x_min = cmin(w_index - half_window)
            x_max = cmax(w_index + half_window, width)
            diff = np.square(resampling_grid[0:batchsize, y_min:y_max, x_min:x_max, 0] - np.tile(identity[0:batchsize, h_index, w_index, 0],
                    (1, y_max - y_min, x_max - x_min))) + np.square(resampling_grid[0:batchsize, y_min:y_max, x_min:x_max, 1] -
                    np.tile(identity[0:batchsize, h_index, w_index, 1], (1, y_max - y_min, x_max - x_min)))
            for b in range(batchsize):
                index = diff[b, :, :].argmin()
                re_h = index // (x_max - x_min) + y_min
                re_w = index - index // (x_max - x_min) * (x_max - x_min) + x_min
                defor_re[b, h_index, w_index, 0] = - defor[b, re_h, re_w, 0]
                defor_re[b, h_index, w_index, 1] = - defor[b, re_h, re_w, 1]
    return defor_re

# @METRIC_REGISTRY.register()
# def calculate_MRE(src, tgt, **kwargs):
# MRE平均重投影误差其实就是TRE的一种

# 也可以叫特征重投影误差或点对齐误差？
def compute_tre(points_1, points_2):
    """ computing Target Registration Error for each landmark pair

    :param ndarray points_1: set of points
    :param ndarray points_2: set of points
    :return ndarray: list of errors of size min nb of points
    array([ 0.21...,  0.70...,  0.44...,  0.34...,  0.41...,  0.41...])
    """
    points_1 = points_1[0, ...]
    points_2 = points_2[0, ...]
    nb_common = min([len(pts) for pts in [points_1, points_2]
                     if pts is not None])
    assert nb_common > 0, 'no common landmarks for metric'
    points_1 = points_1[:nb_common]
    points_2 = points_2[:nb_common]
    diffs = torch.sqrt(torch.sum(torch.square(points_1 - points_2)))
    return diffs

# 又叫变换误差
@METRIC_REGISTRY.register()
def calculate_EPE(pred_flow, gt_flow, **kwargs):
    # flow: [B, H, W, 2]
    inv_gt_flow = defor_reverse(gt_flow)
    inv_gt_flow = torch.from_numpy(inv_gt_flow).to(pred_flow.device)
    # print(torch.isnan(inv_gt_flow).any(), torch.isnan(pred_flow).any())
    # print(inv_gt_flow.max(), inv_gt_flow.min(), pred_flow.max(), pred_flow.min())
    # inv_gt_mse = F.mse_loss(pred_flow, inv_gt_flow)
    inv_gt_l2 = (pred_flow-inv_gt_flow).pow(2).sqrt().mean()
    # gt_mse = F.mse_loss(pred_inv_flow, gt_flow) # 比如pred_flow代表ir2vi(扭曲ir向vi对齐)，那么pred_inv_flow就是vi2ir（扭曲vi向ir对齐），此时网络的flow_direction = 'bi'
    # print("inv_gt_mse   inv_gt_l2   gt_mse")
    # print(inv_gt_mse, inv_gt_l2, gt_mse)
    # import time; time.sleep(10)
    # return inv_gt_mse
    # print(inv_gt_l2)
    return inv_gt_l2  # 对应EPE的定义
    # return (pred_flow-inv_gt_flow).pow(2).sqrt().mean()

class LNCC(nn.Module):
    """
        Local (over window) normalized cross correlation.
    """
    def __init__(self, win=None):
        super(LNCC, self).__init__()
        self.win = win
    def compute_local_sums(self, I, J, filt, stride, padding, win):
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = F.conv2d(I, filt, stride=stride, padding=padding)
        J_sum = F.conv2d(J, filt, stride=stride, padding=padding)
        I2_sum = F.conv2d(I2, filt, stride=stride, padding=padding)
        J2_sum = F.conv2d(J2, filt, stride=stride, padding=padding)
        IJ_sum = F.conv2d(IJ, filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        return I_var, J_var, cross

    def forward(self, I, J):
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        
        win = self.win
        if isinstance(win, int):
            win = [win]
        if win is None:
            win = [9] * ndims
        else:
            win = win * ndims

        sum_filt = torch.ones([1, 1, *win]).cuda()

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        I_var, J_var, cross = self.compute_local_sums(I, J, sum_filt, stride, padding, win)

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(cc)

@METRIC_REGISTRY.register()
def calculate_SWNCC(src, tgt, **kwargs):
    # RGB2Gray
    weights = torch.tensor([0.299, 0.587, 0.114], device=src.device).view(1, 3, 1, 1)
    src = (src * weights).sum(dim=1, keepdim=True)
    tgt = (tgt * weights).sum(dim=1, keepdim=True)
    lncc = LNCC(9)
    _lncc = lncc(src, tgt)
    # print(_lncc)
    return _lncc


# 跟PhotometricLoss一样的思想，逐像素计算配准后图像(ir_warped配准之后的图像)与参考图像（original ir图像）之间的像素值均方误差
@METRIC_REGISTRY.register()
def calculate_PixelMSE(src, tgt, **kwargs):
    mse = F.mse_loss(src, tgt)
    # print(mse)
    return mse