from .mscan import *

from basicsr.utils.registry import ARCH_REGISTRY

import torch.nn.functional as F

# 光照分量引导的类似于cross attention的纯卷积注意力模块
# class LightGuideModulation(nn.Module):
#     def __init__(self, channels, reduction_ratio=4):
#         """
#         LGM模块 (Light-Guided Modulation)
#         参数:
#             channels: 输入通道数
#             reduction_ratio: 通道压缩比例
#         """
#         super().__init__()
        
#         # 光照分支轻量化处理
#         self.l_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, 
#                      padding=1, groups=channels),  # 深度卷积
#             nn.Conv2d(channels, channels, 1),      # 逐点卷积
#             nn.ReLU6(inplace=True))
        
#         # 交叉调制门生成器
#         self.gate = nn.Sequential(
#             nn.Conv2d(2*channels, channels//reduction_ratio, 1),
#             nn.ReLU6(inplace=True),
#             nn.Conv2d(channels//reduction_ratio, channels, 1),
#             nn.Sigmoid())
        
#         # 反射分支残差连接
#         self.r_conv = nn.Sequential(
#             nn.Conv2d(channels, channels, 3,
#                      padding=1, groups=channels),
#             nn.Conv2d(channels, channels, 1))

#     def forward(self, L, R):
#         """
#         输入:
#             L: 光照分量 [B,C,H,W]
#             R: 反射分量 [B,C,H,W]
#         输出:
#             modulated_R: 调制后的反射分量
#         """
#         # 光照特征提取
#         L_feat = self.l_conv(L)
        
#         # 交叉特征拼接
#         fused = torch.cat([L_feat, R], dim=1)
        
#         # 空间注意力生成
#         attn_map = self.gate(fused)
        
#         # 反射分支残差增强
#         r_feat = self.r_conv(R)
        
#         # 注意力调制
#         modulated_R = r_feat * attn_map + R  # 残差连接
        
#         return modulated_R

class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle=small_settings['embed_dims'][0], n_fea_in=4, n_fea_out=3):
        #   n_fea_middle = 64
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle//2, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(n_fea_middle//2, n_fea_middle, kernel_size=3, padding=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.pw_conv = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c=n_fea_middle,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)  # illumination prior
        input = torch.cat([img,mean_c], dim=1)
        del mean_c

        input = self.conv2(self.conv1(input))
        illu_fea = self.depth_conv(input)
        del input
        illu_map = self.pw_conv(illu_fea)
        return illu_fea, illu_map

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class IG_MHA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        # dim == dim_head * heads
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    # def forward(self, x_in, illu_fea_trans):
    #     """
    #     x_in: [b,h,w,c]         # input_feature
    #     illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
    #     return out: [b,h,w,c]
    #     """
    #     b, h, w, c = x_in.shape
    #     # x = x_in.reshape(b, h * w, c)
    #     x = x_in.view(b, h * w, c)
    #     q_inp = self.to_q(x)  # b,hw,c
    #     k_inp = self.to_k(x)  # b,hw,c
    #     v_inp = self.to_v(x)  # b,hw,c
    #     illu_attn = illu_fea_trans # illu_fea->illu_fea_trans: b,c,h,w -> b,h,w,c

    #     # from einops import rearrange
    #     #  q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
    #                             #  (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
    #     n = h*w
    #     q = q_inp.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
    #     k = k_inp.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()   
    #     v = v_inp.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
    #     illu_attn = illu_attn.flatten(1, 2).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
    #     # b,heads,hw,dim_head

    #     v = v * illu_attn
    #     k = k.transpose(-2, -1).contiguous()
    #     q = F.normalize(q, dim=-2, p=2)
    #     k = F.normalize(k, dim=-1, p=2)
    #     attn = (k @ q)   # A = K^T*Q   (b,heads,dim_head,hw) * (b,heads,hw,dim_head) -> (b,heads,dim_head,dim_head)
    #     attn = attn * self.rescale
    #     attn = attn.softmax(dim=-1)
    #     x = v@attn # b,heads,hw,dim_head
    #     x = x.transpose(1,2).contiguous().view(b,n,c)
    #     out_c = self.proj(x).view(b, h, w, c)
    #     out_p = self.pos_emb(v_inp.view(b, h, w, c).permute(
    #         0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
    #     out = out_c + out_p

    #     return out  # b,h,w,c
    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        # x = x_in.reshape(b, h * w, c)
        x = x_in.view(b, h * w, c)
        q_inp = self.to_q(x)  # b,hw,c
        k_inp = self.to_k(x)  # b,hw,c
        v_inp = self.to_v(x)  # b,hw,c
        illu_attn = illu_fea_trans # illu_fea->illu_fea_trans: b,c,h,w -> b,h,w,c

        # del x

        # from einops import rearrange
        #  q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                #  (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        n = h*w
        q = q_inp.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = k_inp.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()   
        v = v_inp.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        illu_attn = illu_attn.flatten(1, 2).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # b,heads,hw,dim_head          

        v = v * illu_attn
        out_p = self.pos_emb(v_inp.view(b, h, w, c).permute(
            0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        
        del q_inp, k_inp, v_inp, illu_attn  

        k = k.transpose(-2, -1).contiguous()
        q = F.normalize(q, dim=-2, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q)   # A = K^T*Q   (b,heads,dim_head,hw) * (b,heads,hw,dim_head) -> (b,heads,dim_head,dim_head)
        
        # del k, q

        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = v@attn # b,heads,hw,dim_head

        del q, k, v, attn

        x = x.transpose(1,2).contiguous().view(b,n,c)
        out_c = self.proj(x).view(b, h, w, c)

        del x
        
        # out = out_c + out_p

        return out_c + out_p  # b,h,w,c

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1).contiguous()

# Illumination-Guided Attention Block
class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MHA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        # x = x.permute(0, 2, 3, 1).contiguous()
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1).contiguous()) + x
            x = ff(x) + x
        # out = x.permute(0, 3, 1, 2).contiguous()
        # return out
        return x


class LLIE_Encoder(nn.Module):
    # heads_num defaults to 8!
    def __init__(self, backbone_settings = small_settings, in_channels=3):
        super(LLIE_Encoder, self).__init__()        
        mscan = MSCAN(**backbone_settings)
        self.depths_in_stage = backbone_settings['depths'][0]
        self.illumination_estimator = Illumination_Estimator(n_fea_middle=backbone_settings['embed_dims'][0])
        self.conv_bridge = nn.Conv2d(in_channels, backbone_settings['embed_dims'][0], kernel_size=3, padding=1, bias=True)
        self.block1 = mscan.block1
        self.norm1 = mscan.norm1
        self.IGAB1 = IGAB(dim=backbone_settings['embed_dims'][0], dim_head=backbone_settings['embed_dims'][0]//8, heads=8, num_blocks=1)
        self.IGAB2 = IGAB(dim=backbone_settings['embed_dims'][0], dim_head=backbone_settings['embed_dims'][0]//8, heads=8, num_blocks=1)

    def forward(self, vi):
        illu_guide, light_up_map = self.illumination_estimator(vi)
        # illu_guide: b,c=64,h,w; light_up_map: b,c=3,h,w        
        light_up = vi * light_up_map  # I_light_up = R + C, where C is the corruption term to be eliminated in following stages
        del light_up_map
        light_up = self.conv_bridge(light_up)  # b,c=64,h,w
        b, c, h, w = light_up.shape
        # light_up = light_up.flatten(2).transpose(1, 2).contiguous()  # b,hw,c=64
        light_up = self.block1[0](light_up, h, w)
        # light_up = light_up.view(b, h, w, c)
        light_up = light_up.permute(0, 2, 3, 1).contiguous()  # b,h,w,c
        light_up = self.IGAB1(light_up, illu_guide)
        # light_up = light_up.view(b, h * w, c)
        light_up = light_up.permute(0, 3, 1, 2).contiguous()  # b,c,h,w
        light_up = self.block1[1](light_up, h, w)
        # light_up = light_up.view(b, h, w, c)
        light_up = light_up.permute(0, 2, 3, 1).contiguous()  # b,h,w,c
        light_up = self.IGAB2(light_up, illu_guide)
        # Don't forget that the first stage of small setting is 2, not 3
        del illu_guide
        if self.depths_in_stage == 3:
            # light_up = light_up.view(b, h * w, c)
            light_up = light_up.permute(0, 3, 1, 2).contiguous()  # b,c,h,w
            light_up = self.block1[2](light_up, h, w)
            # light_up = light_up.flatten(2).transpose(1, 2).contiguous() 
            light_up = light_up.permute(0, 2, 3, 1).contiguous()  # b,h,w,c
        light_up = self.norm1(light_up)
        # light_up = light_up.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # b,c,h,w
        light_up = light_up.permute(0, 3, 1, 2).contiguous()  # b,c,h,w
        return light_up

class LLIE_Decoder(nn.Module):
    def __init__(self, input_channels=small_settings['embed_dims'][0], output_channels=3):
        super(LLIE_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, 3, 1, 1, bias=True),
            nn.Conv2d(input_channels//2, output_channels, 3, 1, 1, bias=True),
            nn.Tanh()
        )
    def forward(self, x):
        return (self.decoder(x)+1)/2

class IrReconstructionEncoder(nn.Module):
    def __init__(self, backbone_settings = tiny_settings, in_channels=1):
        super(IrReconstructionEncoder, self).__init__()
        mscan = MSCAN(**backbone_settings)
        self.block1 = mscan.block1
        self.norm1 = mscan.norm1
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, backbone_settings['embed_dims'][0]//4, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(backbone_settings['embed_dims'][0]//4),
            nn.GELU(),
            nn.Conv2d(backbone_settings['embed_dims'][0]//4, backbone_settings['embed_dims'][0], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(backbone_settings['embed_dims'][0])
        )        

    def forward(self, ir):
        ir = self.feature_extractor(ir)
        b, c, h, w = ir.shape
        # ir = ir.flatten(2).transpose(1, 2).contiguous()
        for blk in self.block1:
            ir = blk(ir, h, w)
        ir = ir.permute(0, 2, 3, 1).contiguous()  # b,h,w,c
        ir = self.norm1(ir)
        # ir = ir.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        ir = ir.permute(0, 3, 1, 2).contiguous()  # b,c,h,w
        return ir

class IrReconstructionDecoder(nn.Module):
    def __init__(self, input_channels=tiny_settings['embed_dims'][0], output_channels=1):
        super(IrReconstructionDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//4, 3, 1, 1, bias=True),
            nn.Conv2d(input_channels//4, output_channels, 3, 1, 1, bias=True),
            nn.Tanh()
        )
    def forward(self, x):
        return (self.decoder(x)+1)/2


@ARCH_REGISTRY.register()
class LLIE_VI(nn.Module):
    def __init__(self, backbone_settings = 'small'):
        super(LLIE_VI, self).__init__()
        if backbone_settings == 'small':
            self.backbone_settings = small_settings 
        elif backbone_settings == 'tiny':
            self.backbone_settings = tiny_settings
        elif backbone_settings == 'base':
            self.backbone_settings = base_settings
        elif backbone_settings == 'hyc_small':
            self.backbone_settings = hyc_small_settings
        # elif backbone_settings == 'hyc_tiny':
        #     self.backbone_settings = hyc_tiny_settings
        self.vi_encoder = LLIE_Encoder(backbone_settings=self.backbone_settings)
        self.vi_decoder = LLIE_Decoder(input_channels=self.backbone_settings['embed_dims'][0])

    def forward(self, vi):
        vi_out = self.vi_decoder(self.vi_encoder(vi))
        return vi_out

@ARCH_REGISTRY.register()
class LLIE_IR(nn.Module):
    def __init__(self, backbone_settings = 'tiny'):
        super(LLIE_IR, self).__init__()
        if backbone_settings == 'small':
            self.backbone_settings = small_settings 
        elif backbone_settings == 'tiny':
            self.backbone_settings = tiny_settings
        elif backbone_settings == 'base':
            self.backbone_settings = base_settings
        elif backbone_settings == 'hyc_small':
            self.backbone_settings = hyc_small_settings
        # elif backbone_settings == 'hyc_tiny':
        #     self.backbone_settings = hyc_tiny_settWings
        self.ir_encoder = IrReconstructionEncoder(backbone_settings=self.backbone_settings)
        self.ir_decoder = IrReconstructionDecoder(input_channels=self.backbone_settings['embed_dims'][0])
    def forward(self, ir):
        ir_out = self.ir_decoder(self.ir_encoder(ir))
        return ir_out
