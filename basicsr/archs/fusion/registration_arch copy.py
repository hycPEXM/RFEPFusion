import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

import os

from .mscan import tiny_settings, small_settings, base_settings, hyc_small_settings
from .LLIE_arch import LLIE_Encoder, IrReconstructionEncoder

import kornia.utils as KU
import kornia.filters as KF

class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1]==2:
            disp = disp.permute(0,2,3,1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1],disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=False)

# class PatchUnfold(nn.Module):
#     def __init__(self, kernel_size, stride, padding):
#         super().__init__()
#         self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#     def forward(self, x):
#         b, c, h, w = x.size()
#         padded_x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
#         patches = self.unfold(padded_x)  # (b, c * k^2, num_patches)
#         num_patches = patches.shape[-1]
#         patch_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
#         patch_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
#         patches = patches.transpose(1, 2).reshape(b, num_patches, c, self.kernel_size, self.kernel_size) # (b, num_patches, c, k, k)
#         return patches, (patch_h, patch_w)

# class PatchFold(nn.Module):
#     def __init__(self, kernel_size, stride, padding):
#         super().__init__()
#         self.fold = nn.Fold(kernel_size=kernel_size, stride=stride, padding=padding, output_size=None)
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#     def forward(self, patches, output_size):
#         b, num_patches, c, k_h, k_w = patches.size()
#         patches_flatten = patches.reshape(b, num_patches, c * k_h * k_w).transpose(1, 2) # (b, c * k^2, num_patches)
#         folded = self.fold(patches_flatten, output_size=output_size)
#         return folded

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim_q = dim_q // num_heads
        self.head_dim_k = dim_k // num_heads
        self.head_dim_v = dim_v // num_heads
        self.scale_k = self.head_dim_k ** -0.5

        self.q_proj = nn.Linear(dim_q, dim_q, bias=True)
        self.k_proj = nn.Linear(dim_k, dim_k, bias=True)
        self.v_proj = nn.Linear(dim_v, dim_v, bias=True)

        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim_v, dim_v) # 输出维度应该和 V 的原始维度一致
        self.out_drop = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, N_q, C_q = q.shape
        _, N_k, C_k = k.shape
        _, N_v, C_v = v.shape
        H = self.num_heads

        # Linear projections and reshaping for multi-head
        q_h = self.q_proj(q).reshape(B, N_q, H, self.head_dim_q).permute(0, 2, 1, 3)   # (B, H, N_q, head_dim_q)
        k_h = self.k_proj(k).reshape(B, N_k, H, self.head_dim_k).permute(0, 2, 1, 3)   # (B, H, N_k, head_dim_k)
        v_h = self.v_proj(v).reshape(B, N_v, H, self.head_dim_v).permute(0, 2, 1, 3)   # (B, H, N_v, head_dim_v)

        # Scaled dot-product attention
        attn = (q_h @ k_h.transpose(-2, -1)) * self.scale_k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values and reshaping
        x = (attn @ v_h).transpose(1, 2).reshape(B, N_q, C_v)

        # Output projection
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(), # 或者其他激活函数，如 ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class AttentionInteraction(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, num_heads=8, ffn_hidden_dim=256, dropout=0.0):
        super().__init__()
        self.mha = MultiHeadAttention(dim_q, dim_k, dim_v, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim_q) # 注意这里的norm维度应该和输入q的维度一致
        self.ffn = FeedForward(dim_q, ffn_hidden_dim, dropout) # FFN的输入维度也应该和q的维度一致
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, q, k, v):
        # Multi-Head Attention
        attn_output = self.mha(q, k, v)
        # Add & Norm (residual connection with query q)
        norm1_output = self.norm1(q + attn_output)

        # Feed-Forward Network
        ffn_output = self.ffn(norm1_output)
        # Add & Norm (residual connection)
        output = self.norm2(norm1_output + ffn_output)

        # return output.transpose(1, 2).reshape(output.shape[0], output.shape[2], int(output.shape[1]**0.5), int(output.shape[1]**0.5))
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入通道数和输出通道数不同，使用 1x1 卷积进行匹配
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetFeatureExtractor, self).__init__()
        self.block1 = ResidualBlock(in_channels, out_channels*2)
        self.block2 = ResidualBlock(out_channels*2, out_channels*3)
        self.block3 = ResidualBlock(out_channels*3, out_channels*2)
        self.block4 = ResidualBlock(out_channels*2, out_channels)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)


class CrossModalAttention(nn.Module):
    def __init__(self, in_channels_ir, in_channels_vis):
        super(CrossModalAttention, self).__init__()
        # 示例：使用简单的卷积方式生成注意力
        self.conv_ir = nn.Conv2d(in_channels_ir, 16, kernel_size=3, padding=1)
        self.conv_vis = nn.Conv2d(in_channels_vis, 16, kernel_size=3, padding=1)
        self.conv_attn = nn.Conv2d(32, 1, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Identity()

    def forward(self, x_ir, x_vis):
        feat_ir = self.conv_ir(x_ir)
        feat_vis = self.conv_vis(x_vis)
        feat_concat = torch.cat([feat_ir, feat_vis], dim=1)
        attention_ir = self.sigmoid(self.conv_attn(feat_concat))
        attention_vis = self.sigmoid(self.conv_attn(feat_concat))
        return attention_ir, attention_vis

class CMIA_Conv(nn.Module):
    def __init__(self, in_channels_ir, in_channels_vis, out_channels):
        super(CMIA_Conv, self).__init__()
        self.conv_ir = nn.Conv2d(in_channels_ir, out_channels, kernel_size=3, padding=1)
        self.conv_vis = nn.Conv2d(in_channels_vis, out_channels, kernel_size=3, padding=1)
        self.attention = CrossModalAttention(out_channels, out_channels)
        self.conv_fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1) # 假设使用拼接融合

    def forward(self, x_ir, x_vis):
        feat_ir = self.conv_ir(x_ir)
        feat_vis = self.conv_vis(x_vis)
        attn_ir, attn_vis = self.attention(feat_ir, feat_vis)
        attended_ir = feat_ir * attn_ir
        attended_vis = feat_vis * attn_vis
        fused_feat = torch.cat([attended_ir, attended_vis], dim=1)
        output_feat = self.conv_fuse(fused_feat)
        return output_feat

@ARCH_REGISTRY.register()
class RFEPFusion_reg_net(nn.Module):
    def __init__(self, ir_settings = 'tiny', vi_settings = 'hyc_small',
                 ir_encoder_pretrained_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/ir_encoder.pth',
                 vi_encoder_pretrained_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/vi_encoder.pth',
                 dynamic_shape=False,
                #  unfold_ks = 17,
                #  unfold_stride = 8,
                #  unfold_pad = 8,
                 unfold_ks = 21,
                 unfold_stride = 10,
                 unfold_pad = 10,
                 flow_est_iter = 3,
                #  flow_est_iter = 2,
                 feature_dim = 4,
                 ):
        super(RFEPFusion_reg_net, self).__init__()
        if ir_settings == 'small':
            self.ir_settings = small_settings 
        elif ir_settings == 'tiny':
            self.ir_settings = tiny_settings
        elif ir_settings == 'base':
            self.ir_settings = base_settings
        elif ir_settings == 'hyc_small':
            self.ir_settings = hyc_small_settings
        
        if vi_settings == 'small':
            self.vi_settings = small_settings 
        elif vi_settings == 'tiny':
            self.vi_settings = tiny_settings
        elif vi_settings == 'base':
            self.vi_settings = base_settings
        elif vi_settings == 'hyc_small':
            self.vi_settings = hyc_small_settings                
        
        # self.vi_encoder = LLIE_Encoder(backbone_settings=self.vi_settings)
        # self.ir_encoder = IrReconstructionEncoder(backbone_settings=self.ir_settings)
        
        # if not os.path.exists(vi_encoder_pretrained_path):
        #     raise FileNotFoundError('LLIE_vi_encoder pretrained weights not found')
        # if not os.path.exists(ir_encoder_pretrained_path):
        #     raise FileNotFoundError('LLIE_ir_encoder pretrained weights not found')                        
        
        # self.vi_encoder.load_state_dict(torch.load(vi_encoder_pretrained_path)['params'], strict=True)        
        # self.ir_encoder.load_state_dict(torch.load(ir_encoder_pretrained_path)['params'], strict=True)        
        
        # for param in self.vi_encoder.parameters(): 
        #     param.requires_grad = False
        # for param in self.ir_encoder.parameters():
        #     param.requires_grad = False
        
        # self.ir_encoder = nn.Sequential(LLIE_Encoder(backbone_settings=self.vi_settings).load_state_dict(torch.load(vi_encoder_pretrained_path)['params'], strict=True), nn.Conv2d(self.vi_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1))
        # self.ir_encoder = LLIE_Encoder(backbone_settings=self.vi_settings)
        # self.ir_encoder.load_state_dict(torch.load(vi_encoder_pretrained_path)['params'], strict=True)
        
        
        
        # exp1
        # self.ir_encoder = nn.Sequential(self.vi_encoder, nn.Conv2d(self.vi_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1))
        
        # exp2
        # self.ir_encoder = nn.Conv2d(self.vi_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1)
        
        # exp3
        # self.ir_encoder = LLIE_Encoder(backbone_settings=self.vi_settings)
        # self.ir_encoder.load_state_dict(torch.load(vi_encoder_pretrained_path)['params'], strict=True)                
        # for param in self.ir_encoder.parameters(): 
        #     param.requires_grad = False
        # self.ir_encoder_2 = nn.Conv2d(self.vi_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1)
        
        self.dynamic_shape = (not self.training) and dynamic_shape
        self.unfold_ks = unfold_ks
        self.unfold_stride = unfold_stride
        self.unfold_pad = unfold_pad
        self.flow_est_iter = flow_est_iter
        self.feature_dim = feature_dim
        
        # self.vi_proj_conv = nn.Sequential(
        #     nn.Conv2d(self.vi_settings['embed_dims'][0], self.vi_settings['embed_dims'][0]//2, kernel_size=3, stride=1,padding=1, bias=False),
        #     nn.BatchNorm2d(self.vi_settings['embed_dims'][0]//2),
        #     # nn.GELU(),
        #     nn.Conv2d(self.vi_settings['embed_dims'][0]//2, feature_dim, kernel_size=3, stride=1,padding=1)
        # )
        # self.ir_proj_conv = nn.Sequential(
        #     nn.Conv2d(self.ir_settings['embed_dims'][0], self.ir_settings['embed_dims'][0]//2, kernel_size=3, stride=1,padding=1, bias=False),
        #     nn.BatchNorm2d(self.ir_settings['embed_dims'][0]//2),
        #     # nn.GELU(),
        #     nn.Conv2d(self.ir_settings['embed_dims'][0]//2, feature_dim, kernel_size=3, stride=1, padding=1),
        # )
        
        
        self.ir_encoder = ResNetFeatureExtractor(3, self.vi_settings['embed_dims'][0])
        self.vi_encoder = ResNetFeatureExtractor(3, self.vi_settings['embed_dims'][0])
        self.ir_vi_share_1 = nn.Sequential(
            nn.Conv2d(self.vi_settings['embed_dims'][0], self.vi_settings['embed_dims'][0]//2, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(self.vi_settings['embed_dims'][0]//2),
            nn.GELU(),
            nn.Conv2d(self.vi_settings['embed_dims'][0]//2, feature_dim, kernel_size=3, stride=2, padding=1)
        )
        
        self.ir_vi_share_2 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )
        
        # Spatial Transformer
        # self.register_buffer('grid', None)
        # self.register_buffer('cnt_mat', None)
        # self.register_buffer('scale', None)
        # self.register_buffer('h', None)
        # self.register_buffer('w', None)
        # self.grid = torch.ones((2,2))
        # self.cnt_mat = torch.ones((2,2))
        
        self.register_buffer('grid', None)
        self.register_buffer('grid_down2', None)
        self.register_buffer('cnt_mat_down2', None)
        self.register_buffer('cnt_mat_down4', None)
        self.register_buffer('scale_down2', None)
        self.register_buffer('scale_down4', None)
        
        self.attn_dim = feature_dim*unfold_ks*unfold_ks
        self.flow_est_ir2vi_1 = AttentionInteraction(self.attn_dim, self.attn_dim, self.attn_dim, num_heads=4, ffn_hidden_dim=512, dropout=0.)
        self.flow_est_ir2vi_2 = AttentionInteraction(self.attn_dim, self.attn_dim, self.attn_dim, num_heads=4, ffn_hidden_dim=512, dropout=0.)
        # self.flow_est_vi2ir = AttentionInteraction(self.attn_dim, self.attn_dim, self.attn_dim, num_heads=8, ffn_hidden_dim=256, dropout=0.)        
        # self.deformation_head = nn.Conv2d(2*feature_dim, 2, kernel_size=3, padding=1)
        
        # self.flow_est_ir2vi_1 = CMIA_Conv(self.attn_dim, self.attn_dim, self.attn_dim)
        # self.flow_est_ir2vi_2 = CMIA_Conv(self.attn_dim, self.attn_dim, self.attn_dim)
        self.deformation_head_1 = nn.Conv2d(feature_dim, 2, kernel_size=3, padding=1)
        self.deformation_head_2 = nn.Conv2d(feature_dim, 2, kernel_size=3, padding=1)

        print(f"Param of '{self.__class__.__name__}': {sum(p.numel() for p in self.parameters() if p.requires_grad)/1024/1024} M")
        
    def generate_grid_mat(self, ir, vi):
        self.h = ir.shape[2]
        self.w = ir.shape[3]
        self.grid = KU.create_meshgrid(self.h, self.w)
        self.grid = self.grid.type(torch.FloatTensor).to(next(self.parameters()).device)
        
        
        # 一定注意设置stride、ks、pad，否则cnt_mat会出现0值，表示原张量某些位置的元素没有经过滑窗计算，导致除零NAN值
        # unfold_ks = 21,
        #          unfold_stride = 12,
        #          unfold_pad = 10,
        # 对于高度480而言
        # num_win = (480 + 2*10 - (21-1)-1)/12 + 1注意是向下取整，结果为40，也就是经过40个滑窗
        # (40-1)*12 + 21 = 489 < 480 + 10 = 490  最后一个滑窗的下边界只到489，小于上边pad后的原图下边界，证明最后一个元素没有被滑窗到，对应展开fold之后该位置为0
        B,C,H,W = ir.shape
        ones_tensor = torch.ones(1,self.feature_dim,H,W)
        unfolded_ones = F.unfold(ones_tensor, kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)
        self.cnt_mat = F.fold(unfolded_ones, output_size=ir.shape[2:], kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride).to(self.grid.device)
        # print(self.cnt_mat)
        
        # self.scale = (torch.FloatTensor([W,H]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1).to(self.grid.device)
        
    def generate_grid_mat_(self, ir):
        self.h = ir.shape[2]
        self.w = ir.shape[3]
        h_down2, w_down2 = (self.h+1)//2, (self.w+1)//2
        self.grid_down2 = KU.create_meshgrid(h_down2, w_down2)
        self.grid_down2 = self.grid_down2.type(torch.FloatTensor).to(next(self.parameters()).device)

        self.grid = KU.create_meshgrid(self.h, self.w)
        self.grid = self.grid.type(torch.FloatTensor).to(next(self.parameters()).device)
        
        ones_tensor = torch.ones(1,self.feature_dim,h_down2,w_down2).to(self.grid_down2.device)
        unfolded_ones = F.unfold(ones_tensor, kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)
        self.cnt_mat_down2 = F.fold(unfolded_ones, output_size=(h_down2, w_down2), kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)
        
        h_down4, w_down4 = (h_down2+1)//2, (w_down2+1)//2
        ones_tensor = torch.ones(1,self.feature_dim,h_down4,w_down4).to(self.grid_down2.device)
        unfolded_ones = F.unfold(ones_tensor, kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)
        self.cnt_mat_down4 = F.fold(unfolded_ones, output_size=(h_down4, w_down4), kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)       
        
        self.scale_down2 =  (torch.FloatTensor([w_down2,h_down2]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1).to(self.grid_down2.device)
        self.scale_down4 =  (torch.FloatTensor([w_down4,h_down4]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1).to(self.grid_down2.device)
        
    
    def generate_mask(self, disp):
        assert self.grid != None
        flow = self.grid + disp
        goodmask = torch.logical_and(flow >= -1, flow <= 1)
        goodmask = torch.logical_and(goodmask[..., 0], goodmask[..., 1]).unsqueeze(1) * 1.0
        AP = nn.AvgPool2d(5, stride=1, padding=2)
        for _ in range(2):
            goodmask = (AP(goodmask) > 0.3).float()
        
        flow = self.grid - disp
        goodmask_inv = F.grid_sample(goodmask, flow, align_corners=False)
        return goodmask, goodmask_inv
    
    def spatial_transform(self, img, disp):               
        disp = disp.permute(0,2,3,1)
        # _,_,h,w = img.size() 
        # disp[..., 0] = 2*disp[..., 0]/max(w-1, 1) - 1.0
        # disp[..., 1] = 2*disp[..., 1]/max(h-1, 1) - 1.0    
        flow = self.grid + disp
        return F.grid_sample(img, flow, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # cross-modal registration
    def CMregister(self, ir, vi):
        # if self.dynamic_shape:
        #     self.generate_grid_mat(ir, vi)
        # elif not (hasattr(self, 'h') or hasattr(self,'w')):
        #     self.generate_grid_mat(ir, vi)
        # elif (self.h != ir.shape[2] or self.w != ir.shape[3]):
        #     self.generate_grid_mat(ir, vi)
        
        if not (hasattr(self, 'h') or hasattr(self,'w')):
            self.generate_grid_mat_(ir)
        elif (self.h != ir.shape[2] or self.w != ir.shape[3]):
            self.generate_grid_mat_(ir)
        
        B, C, H, W = ir.shape
        # print('vi:',vi.shape)
        # print('ir:',ir.shape)
        # print(self.cnt_mat.shape)
        
        ir_ori_img = ir.clone()
        
        vi = self.vi_encoder(vi) 
        
        # 三次实验都没有解决：INFO:torch.nn.parallel.distributed:Reducer buckets have been rebuilt in this iteration.
        # # exp1 
        ir = self.ir_encoder(ir)
        
        # # exp2
        # ir = self.vi_encoder(ir)
        # ir = self.ir_encoder(ir)
        
        # exp3
        # ir = self.ir_encoder(ir)
        # ir = self.ir_encoder_2(ir)
        
        # vi = self.vi_proj_conv(vi)
        # ir = self.ir_proj_conv(ir)                
        
        ir_down2, vi_down2 = torch.split(self.ir_vi_share_1(torch.cat([ir,vi])),B)
        ir_down4, vi_down4 = torch.split(self.ir_vi_share_2(torch.cat([ir_down2,vi_down2])),B)
        
        # 多尺度方案!!!!!!!!!!!!!!!!!!!!!!!!!
        patches_vi = F.unfold(vi_down4, kernel_size=self.unfold_ks,
                                padding=self.unfold_pad, stride=self.unfold_stride)
        num_patches = patches_vi.shape[-1]
        patches_vi = patches_vi.transpose(1, 2).reshape(B,-1,self.unfold_ks,self.unfold_ks)
        patches_ir = F.unfold(ir_down4, kernel_size=self.unfold_ks,
                                padding=self.unfold_pad, stride=self.unfold_stride).transpose(1, 2).reshape(B,-1,self.unfold_ks,self.unfold_ks)        
        attn_folded = self.flow_est_ir2vi_1(patches_ir, patches_vi, patches_vi).reshape(B, num_patches, -1).transpose(1,2)  
        # 记得除以cnt_mat
        attn_folded = F.fold(attn_folded, output_size=ir_down4.shape[2:], kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)
        attn_folded = attn_folded/self.cnt_mat_down4
        
        deformation_field_down4 = self.deformation_head_1(attn_folded)
        deformation_field_down4 = KF.gaussian_blur2d(deformation_field_down4, (13,13), (3,3), border_type='replicate')/self.scale_down4
        deformation_field_down4 = F.interpolate(deformation_field_down4, ir_down2.shape[2:], mode='bilinear', align_corners=False)
        
        ir_down2 = F.grid_sample(ir_down2, self.grid_down2+deformation_field_down4.permute(0,2,3,1), align_corners=False)
        
        patches_vi = F.unfold(vi_down2, kernel_size=self.unfold_ks,
                                padding=self.unfold_pad, stride=self.unfold_stride)
        num_patches = patches_vi.shape[-1]
        patches_vi = patches_vi.transpose(1, 2).reshape(B,-1,self.unfold_ks,self.unfold_ks)
        patches_ir = F.unfold(ir_down2, kernel_size=self.unfold_ks,
                                padding=self.unfold_pad, stride=self.unfold_stride).transpose(1, 2).reshape(B,-1,self.unfold_ks,self.unfold_ks)        
        attn_folded = self.flow_est_ir2vi_1(patches_ir, patches_vi, patches_vi).reshape(B, num_patches, -1).transpose(1,2)      
        # 记得除以cnt_mat
        attn_folded = F.fold(attn_folded, output_size=ir_down2.shape[2:], kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)
        attn_folded = attn_folded/self.cnt_mat_down2
        
        deformation_field_down2 = self.deformation_head_2(attn_folded)
        deformation_field_down2 = KF.gaussian_blur2d(deformation_field_down2, (13,13), (3,3), border_type='replicate')/self.scale_down2
        deformation_field_down2 = F.interpolate(deformation_field_down2, (H,W), mode='bilinear', align_corners=False)
        deformation_field_down4 = F.interpolate(deformation_field_down4, (H,W), mode='bilinear', align_corners=False)
        
        deformation_field = deformation_field_down2 + deformation_field_down4
        
        return self.spatial_transform(ir_ori_img, deformation_field), deformation_field.permute(0,2,3,1)
               
        
        # # ir_warped = ir 
        # deformation_field = torch.zeros(B, 2, H, W).to(ir.device)
        # for _ in range(self.flow_est_iter):
        #     ir_warped = self.spatial_transform(ir, deformation_field)
        #     patches_ir = F.unfold(ir_warped, kernel_size=self.unfold_ks,
        #                           padding=self.unfold_pad, stride=self.unfold_stride).transpose(1, 2)            
        #     # num_patches = patches_ir.shape[1]  # 对应NLP中的seq_len
            
        #     # attn_ir2vi = self.flow_est_ir2vi(patches_ir, patches_vi, patches_vi)
        #     # attn_vi2ir = self.flow_est_vi2ir(patches_vi, patches_ir, patches_ir)            
        #     # attn_folded = torch.cat([attn_ir2vi, attn_vi2ir], dim=-1).permute(0,2,1)
            
        #     # attn_folded = self.flow_est_ir2vi(patches_ir, patches_vi, patches_vi).permute(0,2,1)
        #     attn_folded = self.flow_est_ir2vi_1(patches_ir, patches_vi, patches_vi)
        #     attn_folded = self.flow_est_ir2vi_2(attn_folded, patches_vi, patches_vi).permute(0,2,1)
        #     # 记得除以cnt_mat
        #     attn_folded = F.fold(attn_folded, output_size=(H,W), kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)
        #     # print(attn_folded.shape, self.cnt_mat.shape)
        #     attn_folded = attn_folded/self.cnt_mat
        #     # print((self.cnt_mat==0).sum(), self.cnt_mat.shape)
            
        #     deformation_increment = self.deformation_head(attn_folded)
            
        #     # 刚开始训练时sigma设置成21，模糊效果太大，导致基本没有deform？
        #     deformation_increment = KF.gaussian_blur2d(deformation_increment, (13,13), (3,3), border_type='replicate')
                        
        #     deformation_field += deformation_increment/self.scale # (B,2,H,W)         
        
        # 不使用迭代，且去掉了配准部分的激活函数，因为认为激活函数在负数区域会压缩，导致deformation_field的强制变形
        # patches_ir = F.unfold(ir, kernel_size=self.unfold_ks,
        #                         padding=self.unfold_pad, stride=self.unfold_stride).transpose(1, 2)            
        # attn_folded = self.flow_est_ir2vi_1(patches_ir, patches_vi, patches_vi)        
        # # 记得除以cnt_mat
        # attn_folded = F.fold(attn_folded, output_size=(H,W), kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)
        # attn_folded = attn_folded/self.cnt_mat
        
        # deformation_field = self.deformation_head(attn_folded)
        
        # # 刚开始训练时sigma设置成21，模糊效果太大，导致基本没有deform？
        # deformation_field = KF.gaussian_blur2d(deformation_field, (13,13), (3,3), border_type='replicate')/self.scale
        
        # ir_warped = self.spatial_transform(ir, deformation_field)
        # patches_ir = F.unfold(ir_warped, kernel_size=self.unfold_ks,
        #                         padding=self.unfold_pad, stride=self.unfold_stride).transpose(1, 2)            
        # attn_folded = self.flow_est_ir2vi_2(patches_ir, patches_vi, patches_vi)        
        # # 记得除以cnt_mat
        # attn_folded = F.fold(attn_folded, output_size=(H,W), kernel_size=self.unfold_ks, padding=self.unfold_pad, stride=self.unfold_stride)
        # attn_folded = attn_folded/self.cnt_mat
        
        # deformation_field = self.deformation_head(attn_folded)
        
        # # 刚开始训练时sigma设置成21，模糊效果太大，导致基本没有deform？
        # deformation_field = KF.gaussian_blur2d(deformation_field, (13,13), (3,3), border_type='replicate')/self.scale                
        
        # # return {'ir_warped': self.spatial_transform(ir_ori_img, deformation_field),
        # #         'disp_field': deformation_field.permute(0,2,3,1)}
        # return self.spatial_transform(ir_ori_img, deformation_field), deformation_field.permute(0,2,3,1)
    
    def forward(self, ir, vi,  flow_direction='ir2vi'):
        # 不是这里的条件语句导致INFO:torch.nn.parallel.distributed:Reducer buckets have been rebuilt in this iteration.
        if flow_direction == 'bi':
            # ir_pred, ir_disp_pred, vi_pred, vi_disp_pred = self.net_g(ir_stack, vi_stack, flow_direction='bi') 
            ir_pred, ir_disp_pred = self.CMregister(ir, vi)
            vi_pred, vi_disp_pred = self.CMregister(vi, ir)
            return ir_pred, ir_disp_pred, vi_pred, vi_disp_pred
        else:
            return self.CMregister(ir, vi)
        # ir_pred, ir_disp_pred = self.CMregister(ir, vi)
        # vi_pred, vi_disp_pred = self.CMregister(vi, ir)
        # return ir_pred, ir_disp_pred, vi_pred, vi_disp_pred
        