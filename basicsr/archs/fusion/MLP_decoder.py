import numpy as np
import torch.nn as nn
import torch

# from torch.nn.modules import module
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding: 
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    # def __init__(self,
    #              in_channels=[64, 128, 320, 512],
    #              num_classes=40,
    #              dropout_ratio=0.1,
    #              norm_layer=nn.BatchNorm2d,
    #              embed_dim=768,
    #              align_corners=False):
    def __init__(self,
                 in_channels=[128, 320, 512],
                 num_classes=9,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                #  embed_dim=512,
                # embed_dim = [256, 480, 640],
                 embed_dim = [960, 864, 768],                
                 align_corners=False):    
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        out_embed_dim = embed_dim[-1]
        
        self.linear_c = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.linear_c.append(MLP(input_dim=self.in_channels[i], embed_dim=embed_dim[i]))
        
        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=sum(embed_dim), out_channels=out_embed_dim, kernel_size=1, bias=False),
                            norm_layer(out_embed_dim),
                            nn.GELU()
                            )
                            
        self.linear_pred = nn.Conv2d(out_embed_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # 1/2, 1/4, 1/8
        _c = []
        n = inputs[0].shape[0]
        _c.append(self.linear_c[0](inputs[0]).permute(0,2,1).reshape(n, -1, inputs[0].shape[2], inputs[0].shape[3]).contiguous())
        for i in range(1, len(self.in_channels)):
            c = self.linear_c[i](inputs[i]).permute(0,2,1).reshape(n, -1, inputs[i].shape[2], inputs[i].shape[3]).contiguous()
            _c.append(F.interpolate(c, size=inputs[0].size()[2:],mode='bilinear',align_corners=self.align_corners))
        
        _c = self.linear_fuse(torch.cat(_c, dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
