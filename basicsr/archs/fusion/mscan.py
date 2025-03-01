import torch
import torch.nn as nn
# import math
# import warnings
from torch.nn.modules.utils import _pair as to_2tuple

# from mmseg.models.builder import BACKBONES

# from mmcv.cnn import build_norm_layer
# from mmcv.runner import BaseModule
# from mmcv.cnn.bricks import DropPath
# from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
#                                         trunc_normal_init)
# from timm.models.layers import DropPath
from timm.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            norm_layer(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            norm_layer(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        # _, _, H, W = x.size()
        # x = x.flatten(2).transpose(1, 2).contiguous()
        # return x, H, W
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        # return attn * u
        return attn * x


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H=None, W=None):
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        # x = x.view(B, C, N).permute(0, 2, 1).contiguous()
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        # x = x.flatten(2).transpose(1, 2).contiguous()

        return x, H, W



class MSCAN(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[32, 64, 160, 256],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 depths=[3, 3, 5, 2],
                 num_stages=4,
                 norm_layer=nn.BatchNorm2d):
        super(MSCAN, self).__init__()        

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(in_chans, embed_dims[0], norm_layer=norm_layer)
            else:
                patch_embed = OverlapPatchEmbed(patch_size=3,
                                                stride=2,
                                                in_chans=embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_layer=norm_layer)

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_layer=norm_layer)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x):
        pass    


class DWConv(nn.Module):
    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

tiny_settings = {
    'in_chans': 3,
    'embed_dims': [32, 64, 160, 256],
    'mlp_ratios': [8, 8, 4, 4],
    'drop_rate': 0.,
    'drop_path_rate': 0.1,
    'depths': [3, 3, 5, 2],
    'num_stages': 4,
    'norm_layer': nn.BatchNorm2d
}

# hyc_tiny_settings = {
#     'in_chans': 3,
#     'embed_dims': [32, 64, 160, 256],
#     'mlp_ratios': [4, 4, 4, 4],
#     'drop_rate': 0.,
#     'drop_path_rate': 0.1,
#     'depths': [3, 3, 6, 3],
#     'num_stages': 4,
#     'norm_layer': nn.BatchNorm2d
# }

small_settings = {
    'in_chans': 3,
    'embed_dims': [64, 128, 320, 512],        
    'mlp_ratios': [8, 8, 4, 4],
    'drop_rate': 0.,
    'drop_path_rate': 0.1,
    'depths': [2, 2, 4, 2],
    'num_stages': 4,
    'norm_layer': nn.BatchNorm2d
}

hyc_small_settings = {
    'in_chans': 3,
    'embed_dims': [64, 128, 320, 512],        
    'mlp_ratios': [6, 6, 3, 3],
    'drop_rate': 0.,
    'drop_path_rate': 0.1,
    'depths': [3, 3, 5, 2],
    'num_stages': 4,
    'norm_layer': nn.BatchNorm2d
}

# hyc_small_settings = {
#     'in_chans': 3,
#     'embed_dims': [64, 128, 320, 512],        
#     'mlp_ratios': [4, 4, 3, 3],
#     'drop_rate': 0.,
#     'drop_path_rate': 0.1,
#     'depths': [3, 3, 5, 2],
#     'num_stages': 4,
#     'norm_layer': nn.BatchNorm2d
# }

# hyc_small_settings = {
#     'in_chans': 3,
#     'embed_dims': [64, 128, 320, 512],        
#     'mlp_ratios': [6, 6, 4, 4],
#     'drop_rate': 0.,
#     'drop_path_rate': 0.1,
#     'depths': [3, 3, 6, 3],
#     'num_stages': 4,
#     'norm_layer': nn.BatchNorm2d
# }

base_settings = {
    'in_chans': 3,
    'embed_dims': [64, 128, 320, 512],        
    'mlp_ratios': [8, 8, 4, 4],
    'drop_rate': 0.,
    'drop_path_rate': 0.1,
    'depths': [3, 3, 12, 3],
    'num_stages': 4,
    'norm_layer': nn.BatchNorm2d
}