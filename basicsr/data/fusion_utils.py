import torch
# import torch.nn.functional as F
import numpy as np


"""
Define task metrics, loss functions and model trainer here.
"""
def randrot(img):
    mode = np.random.randint(0,4)
    return rot(img,mode)

def randfilp(img):
    mode = np.random.randint(0,3)
    return flip(img,mode)

def rot(img, rot_mode):
    if rot_mode == 0:
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:
        img = img.flip(-2)
        img = img.transpose(-2, -1)
    return img

def flip(img, flip_mode):
    if flip_mode == 0:
        img = img.flip(-2)
    elif flip_mode == 1:
        img = img.flip(-1)
    return img

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out
