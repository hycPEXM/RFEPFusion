import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, constant_


def init_weights(module):
    """初始化模型权重的函数
    
    Args:
        module: 需要初始化的模块
        
    注意:
        - nn.Conv2d 和 nn.Linear 使用 trunc_normal_ 初始化
        - LayerNorm、BatchNorm 和 GroupNorm 使用 constant_ 初始化
        - 只对有可学习参数的标准化层进行初始化
    
    Usage:
        model = MSCAN(**mscan_settings)
        model.apply(init_weights)
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            constant_(module.bias, 0)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        if hasattr(module, 'weight') and module.weight is not None:
            constant_(module.weight, 1.0)
        if hasattr(module, 'bias') and module.bias is not None:
            constant_(module.bias, 0)

class UpsampleConv(nn.Module):
    """上采样卷积模块
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        scale (int): 上采样倍数，可选 2 或 4
        
    注意:
        - scale=2 时使用一次转置卷积进行2倍上采样
        - scale=4 时使用两次转置卷积进行4倍上采样
    """
    def __init__(self, in_channels, out_channels, scale=2, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU):
        super(UpsampleConv, self).__init__()
        if scale == 2:
            self.upsample = nn.ConvTranspose2d(
                in_channels, 
                out_channels,
                kernel_size=2,
                stride=2
            )            
        elif scale == 4:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2,
                    bias=False
                ),
                norm_layer(out_channels),
                act_layer(),
                nn.ConvTranspose2d(
                    out_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2,
                    bias=False
                ),
                norm_layer(out_channels)                
            )
        else:
            raise ValueError(f"不支持的上采样倍数: {scale}, 只支持 2 或 4")
            
    def forward(self, x):
        return self.upsample(x)