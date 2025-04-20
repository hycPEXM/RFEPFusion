import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

import os

from .mscan import StemConv, tiny_settings, small_settings, base_settings, hyc_small_settings, MSCAN
from .LLIE_arch import LLIE_Encoder, IrReconstructionEncoder
from .fusion_modules import CSA_CMR_Module, CMAF_Module, SIM, MultiScaleFusion
from .seg_decoder import *
from .net_utils import *
from .MLP_decoder import DecoderHead as MLPHead
# from .UPerNet_decoder import UPerHead

# 用矩阵分解的老版本
# Asymmetric dual-stream/dual-path backbone (using backbone variant of different sizes for different modalities)
# symmetric fusion modules
# @ARCH_REGISTRY.register()
# class RFEPFusion_no_register(nn.Module):
#     def __init__(self, ir_settings = 'tiny', vi_settings = 'small', stem_scale = 2, 
#                  ir_encoder_pretrained_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/ir_encoder.pth',
#                  vi_encoder_pretrained_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/vi_encoder.pth'):
#         super(RFEPFusion_no_register, self).__init__()
#         if ir_settings == 'small':
#             self.ir_settings = small_settings 
#         elif ir_settings == 'tiny':
#             self.ir_settings = tiny_settings
#         elif ir_settings == 'base':
#             self.ir_settings = base_settings
#         elif ir_settings == 'hyc_small':
#             self.ir_settings = hyc_small_settings
        
#         if vi_settings == 'small':
#             self.vi_settings = small_settings 
#         elif vi_settings == 'tiny':
#             self.vi_settings = tiny_settings
#         elif vi_settings == 'base':
#             self.vi_settings = base_settings
#         elif vi_settings == 'hyc_small':
#             self.vi_settings = hyc_small_settings
        
#         vi_backbone = MSCAN(**self.vi_settings)
#         ir_backbone = MSCAN(**self.ir_settings)
        
#         self.vi_encoder = LLIE_Encoder(backbone_settings=self.vi_settings)
#         self.ir_encoder = IrReconstructionEncoder(backbone_settings=self.ir_settings)
        
#         if not os.path.exists(vi_encoder_pretrained_path):
#             raise FileNotFoundError('LLIE_vi_encoder pretrained weights not found')
#         if not os.path.exists(ir_encoder_pretrained_path):
#             raise FileNotFoundError('LLIE_ir_encoder pretrained weights not found')        
        
#         if stem_scale == 2:
#             self.stem_ir = nn.Conv2d(in_channels=self.ir_settings['embed_dims'][0], 
#                                      out_channels=self.ir_settings['embed_dims'][0], 
#                                      kernel_size=3, 
#                                      stride=2, 
#                                      padding=1)
#             self.stem_vi = nn.Conv2d(in_channels=self.vi_settings['embed_dims'][0], 
#                                      out_channels=self.vi_settings['embed_dims'][0], 
#                                      kernel_size=3, 
#                                      stride=2, 
#                                      padding=1)
#         elif stem_scale == 4:
#             self.stem_ir = StemConv(in_channels=self.ir_settings['embed_dims'][0], 
#                                     out_channels=self.ir_settings['embed_dims'][0])
#             self.stem_vi = StemConv(in_channels=self.vi_settings['embed_dims'][0], 
#                                     out_channels=self.vi_settings['embed_dims'][0])

#         self.blocks_ir = nn.ModuleList(
#             [ir_backbone.block2, ir_backbone.block3, ir_backbone.block4]
#         )
#         self.norms_ir = nn.ModuleList(
#             [ir_backbone.norm2, ir_backbone.norm3, ir_backbone.norm4]
#         )
#         self.patch_embeds_ir = nn.ModuleList(
#             [ir_backbone.patch_embed2, ir_backbone.patch_embed3, ir_backbone.patch_embed4]
#         )

#         self.blocks_vi = nn.ModuleList(
#             [vi_backbone.block2, vi_backbone.block3, vi_backbone.block4]
#         )
#         self.norms_vi = nn.ModuleList(
#             [vi_backbone.norm2, vi_backbone.norm3, vi_backbone.norm4]
#         )
#         self.patch_embeds_vi = nn.ModuleList(
#             [vi_backbone.patch_embed2, vi_backbone.patch_embed3, vi_backbone.patch_embed4]
#         )

#         # self.block2_vi = vi_backbone.block2
#         # self.norm2_vi = vi_backbone.norm2
#         # self.patch_embed2_vi = vi_backbone.patch_embed2
#         # self.block3_vi = vi_backbone.block3
#         # self.norm3_vi = vi_backbone.norm3
#         # self.patch_embed3_vi = vi_backbone.patch_embed3
#         # self.block4_vi = vi_backbone.block4
#         # self.norm4_vi = vi_backbone.norm4
#         # self.patch_embed4_vi = vi_backbone.patch_embed4

#         # self.block2_ir = ir_backbone.block2
#         # self.norm2_ir = ir_backbone.norm2
#         # self.patch_embed2_ir = ir_backbone.patch_embed2
#         # self.block3_ir = ir_backbone.block3
#         # self.norm3_ir = ir_backbone.norm3
#         # self.patch_embed3_ir = ir_backbone.patch_embed3
#         # self.block4_ir = ir_backbone.block4
#         # self.norm4_ir = ir_backbone.norm4
#         # self.patch_embed4_ir = ir_backbone.patch_embed4

#         if self.ir_settings['embed_dims'][0] != self.vi_settings['embed_dims'][0]:
#             # self.conv_align_stage1 = nn.Sequential(
#             #     nn.BatchNorm2d(self.ir_settings['embed_dims'][0]),
#             #     nn.Conv2d(self.ir_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1, bias=False), 
#             #     nn.BatchNorm2d(self.ir_settings['embed_dims'][0]),
#             #     nn.Conv2d(self.ir_settings['embed_dims'][0], self.vi_settings['embed_dims'][0], kernel_size=1, bias=True))
#             # self.conv_dealign_stage1 = nn.Sequential(
#             #     nn.Conv2d(self.vi_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1, bias=False), 
#             #     nn.BatchNorm2d(self.ir_settings['embed_dims'][0]))
#             self.conv_align_stage1 = nn.Conv2d(self.ir_settings['embed_dims'][0], self.vi_settings['embed_dims'][0], kernel_size=1, bias=True)
#             self.conv_dealign_stage1 = nn.Conv2d(self.vi_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1, bias=True)
#         else:
#             self.conv_align_stage1 = nn.Identity()
#             self.conv_dealign_stage1 = nn.Identity()

#         self.conv_align_stage2_4 = nn.ModuleList()
#         self.conv_dealign_stage2_4 = nn.ModuleList()
#         for i in range(1, 4):
#             if self.ir_settings['embed_dims'][i] != self.vi_settings['embed_dims'][i]:
#                 self.conv_align_stage2_4.append(nn.Conv2d(self.ir_settings['embed_dims'][i], self.vi_settings['embed_dims'][i], kernel_size=1, bias=True))
#                 if i != 3:
#                     self.conv_dealign_stage2_4.append(nn.Conv2d(self.vi_settings['embed_dims'][i], self.ir_settings['embed_dims'][i], kernel_size=1, bias=True))
#             else:
#                 self.conv_align_stage2_4.append(nn.Identity())
#                 if i != 3:
#                     self.conv_dealign_stage2_4.append(nn.Identity())
#         # if self.ir_settings['embed_dims'][1] != self.vi_settings['embed_dims'][1]:
#         #     self.conv_align_stage2_4.append(nn.Conv2d(self.ir_settings['embed_dims'][1], self.vi_settings['embed_dims'][1], kernel_size=1, bias=True))
#         #     self.conv_dealign_stage2_4.append(nn.Conv2d(self.vi_settings['embed_dims'][1], self.ir_settings['embed_dims'][1], kernel_size=1, bias=True))
#         # else:
#         #     self.conv_align_stage2_4.append(nn.Identity())
#         #     self.conv_dealign_stage2_4.append(nn.Identity())
#         # if self.ir_settings['embed_dims'][2] != self.vi_settings['embed_dims'][2]:
#         #     self.conv_align_stage2_4.append(nn.Conv2d(self.ir_settings['embed_dims'][2], self.vi_settings['embed_dims'][2], kernel_size=1, bias=True))
#         #     self.conv_dealign_stage2_4.append(nn.Conv2d(self.vi_settings['embed_dims'][2], self.ir_settings['embed_dims'][2], kernel_size=1, bias=True))
#         # else:
#         #     self.conv_align_stage2_4.append(nn.Identity())
#         #     self.conv_dealign_stage2_4.append(nn.Identity())
#         # if self.ir_settings['embed_dims'][3] != self.vi_settings['embed_dims'][3]:
#         #     self.conv_align_stage2_4.append(nn.Conv2d(self.ir_settings['embed_dims'][3], self.vi_settings['embed_dims'][3], kernel_size=1, bias=True))
#         #     self.conv_dealign_stage2_4.append(nn.Conv2d(self.vi_settings['embed_dims'][3], self.ir_settings['embed_dims'][3], kernel_size=1, bias=True))
#         # else:
#         #     self.conv_align_stage2_4.append(nn.Identity())
#         #     self.conv_dealign_stage2_4.append(nn.Identity())        
#         self.CSA_CMR_stage1_fusion = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][0])
#         self.CSA_CMR_stages = nn.ModuleList(
#             [CSA_CMR_Module(dim=self.vi_settings['embed_dims'][1]), 
#              CSA_CMR_Module(dim=self.vi_settings['embed_dims'][2]), 
#              CSA_CMR_Module(dim=self.vi_settings['embed_dims'][3])]
#         )
#         # self.CSA_CMR_stage2 = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][1])
#         # self.CSA_CMR_stage3 = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][2])
#         # self.CSA_CMR_stage4 = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][3])

#         self.cmaf_stage1_fusion = CMAF_Module(dim=self.vi_settings['embed_dims'][0], num_heads=1)
#         # 和LLIE一样，num_heads都取为8
#         self.cmaf_stages = nn.ModuleList(
#             [CMAF_Module(dim=self.vi_settings['embed_dims'][1], num_heads=2), 
#              CMAF_Module(dim=self.vi_settings['embed_dims'][2], num_heads=4), 
#              CMAF_Module(dim=self.vi_settings['embed_dims'][3], num_heads=8)]
#         )
#         # self.cmaf_stage2 = CMAF_Module(dim=self.vi_settings['embed_dims'][1])
#         # self.cmaf_stage3 = CMAF_Module(dim=self.vi_settings['embed_dims'][2])
#         # self.cmaf_stage4 = CMAF_Module(dim=self.vi_settings['embed_dims'][3])

#         # self.cmaf_upsample = nn.ModuleList(
#         #     [
#         #         UpsampleConv(in_channels=self.vi_settings['embed_dims'][1], 
#         #                  out_channels=self.vi_settings['embed_dims'][1],
#         #                  scale = stem_scale,
#         #                  norm_layer=self.vi_settings['norm_layer']), 
#         #         UpsampleConv(in_channels=self.vi_settings['embed_dims'][2], 
#         #                      out_channels=self.vi_settings['embed_dims'][2],
#         #                      scale = stem_scale,
#         #                      norm_layer=self.vi_settings['norm_layer']), 
#         #         UpsampleConv(in_channels=self.vi_settings['embed_dims'][3], 
#         #                      out_channels=self.vi_settings['embed_dims'][3],
#         #                      scale = stem_scale,
#         #                      norm_layer=self.vi_settings['norm_layer']),
#         #     ] 
#         # )
#         self.cmaf_upsample = nn.ModuleList(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) for _ in range(3)
#         )

#         if vi_settings == 'small':
#             self.decode_head_settings = small_settings_decode_head 
#         elif vi_settings == 'tiny':
#             self.decode_head_settings = tiny_settings_decode_head
#         elif vi_settings == 'base':
#             self.decode_head_settings = base_settings_decode_head
#         elif vi_settings == 'hyc_small':
#             self.decode_head_settings = small_settings_decode_head 
        
#         # self.eat_hamburger = SIM(norm_nc = self.vi_settings['embed_dims'][0], seg_nc = self.decode_head_settings['ham_channels'])
#         self.fusion = MultiScaleFusion(in_channels=self.vi_settings['embed_dims'][0], mid_channels=32, out_channels=3, dilation_scales=[1, 2, 4])

#         self.seg_decoder = LightHamHead(**self.decode_head_settings)

#         # 用这个初始化方式反而使训练更不稳定
#         # self.apply(init_weights)
        
#         self.apply(init_weights_temp)
        
#         # 这里直接用的params，没有考虑params_ema
        
#         self.vi_encoder.load_state_dict(torch.load(vi_encoder_pretrained_path)['params'], strict=True)        
#         self.ir_encoder.load_state_dict(torch.load(ir_encoder_pretrained_path)['params'], strict=True)
#         # 冻结LLIE encoder的参数
#         # print(list(self.vi_encoder.parameters())[0])
#         # for param in self.vi_encoder.parameters():   
#         # for param in list(self.vi_encoder.parameters())[:-2]:
#         # for name, param in self.vi_encoder.named_parameters():
#         #     # print(name)
#         #     if not (name.startswith('block1.2')) or not (name.startswith('norm1')):
#         #         param.requires_grad = False
#         for param in self.vi_encoder.parameters(): 
#             param.requires_grad = False
#         # print(list(self.vi_encoder.parameters())[0])
#         for param in self.ir_encoder.parameters():
#             param.requires_grad = False
#         # for param in list(self.ir_encoder.parameters())[:-2]:
#         # for name, param in self.ir_encoder.named_parameters():
#         #     if not (name.startswith('block1.2')) or not (name.startswith('norm1')):
#         #         param.requires_grad = False
    
#     # 写一个计算过程和forward一样的函数，检测每个模块的输出（中间结果）是否正确（是否包含NAN值，打印张量的最大值和最小值）
#     # 把GPU_num设为1，比较好定位错误
#     @torch.no_grad()
#     def forward_debug(self, ir, vi):        
#         def check_tensor(tensor, name):
#             if torch.isnan(tensor).any().item():
#                 # raise RuntimeError(f"{name} contains NaN values!")
#                 print(f"{name} contains NaN values!")
#                 return True
#             elif torch.isinf(tensor).any().item():
#                 # raise RuntimeError(f"{name} contains infinite values!")
#                 print(f"{name} contains infinite values!")
#                 return True
#             return False
#             # else:
#             #     print(f"{name} is valid.")

#         _, _, ori_h, ori_w = ir.shape
#         # with torch.no_grad():
#         vi = self.vi_encoder(vi)  # H, W
#         ir = self.ir_encoder(ir)
#         print("ir:", ir.max(), ir.min())
#         print("vi:", vi.max(), vi.min())
#         check_tensor(vi, "vi_encoder output")
#         check_tensor(ir, "ir_encoder output")

#         ir = self.conv_align_stage1(ir)
#         check_tensor(ir, "conv_align_stage1 output")
#         ir, vi = self.CSA_CMR_stage1_fusion(ir, vi)
#         check_tensor(ir, "CSA_CMR_stage1_fusion output (ir)")
#         check_tensor(vi, "CSA_CMR_stage1_fusion output (vi)")
#         fusion = self.cmaf_stage1_fusion(ir, vi)
#         check_tensor(fusion, "cmaf_stage1_fusion output")

#         ir = self.conv_dealign_stage1(ir)
#         check_tensor(ir, "conv_dealign_stage1 output")
#         ir = self.stem_ir(ir)   # H/2, W/2
#         check_tensor(ir, "stem_ir output")
#         vi = self.stem_vi(vi)
#         check_tensor(vi, "stem_vi output")

#         fused_features_for_seg = []  # [(H/2, W/2), (H/4, W/4), (H/8, W/8)]

#         for i in range(3):
#             ir, h_ir, w_ir = self.patch_embeds_ir[i](ir)
#             check_tensor(ir, f"patch_embeds_ir[{i}] output")
#             vi, h_vi, w_vi = self.patch_embeds_vi[i](vi)
#             check_tensor(vi, f"patch_embeds_vi[{i}] output")

#             for blk in self.blocks_ir[i]:
#                 ir = blk(ir, h_ir, w_ir)
#             check_tensor(ir, f"blocks_ir[{i}] output")

#             for blk in self.blocks_vi[i]:
#                 vi = blk(vi, h_vi, w_vi)
#             check_tensor(vi, f"blocks_vi[{i}] output")

#             ir = ir.permute(0, 2, 3, 1).contiguous()
#             vi = vi.permute(0, 2, 3, 1).contiguous()
#             ir = self.norms_ir[i](ir)
#             vi = self.norms_vi[i](vi)
#             check_tensor(ir, f"norms_ir[{i}] output")
#             check_tensor(vi, f"norms_vi[{i}] output")

#             ir = ir.permute(0, 3, 1, 2).contiguous()
#             vi = vi.permute(0, 3, 1, 2).contiguous()
#             ir = self.conv_align_stage2_4[i](ir)
#             check_tensor(ir, f"conv_align_stage2_4[{i}] output")
#             ir, vi = self.CSA_CMR_stages[i](ir, vi)
#             check_tensor(ir, f"CSA_CMR_stages[{i}] output (ir)")
#             check_tensor(vi, f"CSA_CMR_stages[{i}] output (vi)")

#             fused_features_for_seg.append(self.cmaf_upsample[i](self.cmaf_stages[i](ir, vi)))
#             check_tensor(fused_features_for_seg[-1], f"cmaf_stages[{i}] output")

#             if i != 2:
#                 ir = self.conv_dealign_stage2_4[i](ir)
#                 check_tensor(ir, f"conv_dealign_stage2_4[{i}] output")

#         seg_out, hamburger = self.seg_decoder(fused_features_for_seg)
#         check_tensor(seg_out, "seg_decoder output (seg_out)")
#         check_tensor(hamburger, "seg_decoder output (hamburger)")
#         seg_out = F.interpolate(seg_out, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
#         check_tensor(seg_out, "seg_out resized output")

#         # fusion = self.eat_hamburger(fusion, hamburger)
#         # check_tensor(fusion, "eat_hamburger output")
#         fusion = self.fusion(fusion)
#         if check_tensor(fusion, "fusion output"):
#             raise RuntimeError("fusion output is invalid!")
#         print("我真TM服了")
#         return fusion, seg_out       
    
#     def forward(self, ir, vi):
#         _, _, ori_h, ori_w = ir.shape
#         # with torch.no_grad():
#         vi = self.vi_encoder(vi)  # H, W
#         ir = self.ir_encoder(ir)
#         # print(ir)

#         ir = self.conv_align_stage1(ir)
#         ir, vi = self.CSA_CMR_stage1_fusion(ir, vi)
#         fusion = self.cmaf_stage1_fusion(ir, vi)
#         # print(ir)

#         ir = self.conv_dealign_stage1(ir)
#         ir = self.stem_ir(ir)   # H/2, W/2
#         vi = self.stem_vi(vi)

#         fused_features_for_seg = []  # [(H/2, W/2), (H/4, W/4), (H/8, W/8)]

#         # B = ir.shape[0]

#         # stage1: H/4, W/4
#         # stage2: H/8, W/8
#         # stage3: H/16, W/16
#         for i in range(3):
#             ir, h_ir, w_ir = self.patch_embeds_ir[i](ir)  
#             vi, h_vi, w_vi = self.patch_embeds_vi[i](vi)  
#             # block = self.blocks_ir[i]
#             # ir = block(ir, h_ir, w_ir)
#             # ir = self.blocks_ir[i](ir, h_ir, w_ir)
#             # vi = self.blocks_vi[i](vi, h_vi, w_vi)
#             for blk in self.blocks_ir[i]:
#                 ir = blk(ir, h_ir, w_ir)
#             for blk in self.blocks_vi[i]:             
#                 vi = blk(vi, h_vi, w_vi)
#             ir = ir.permute(0, 2, 3, 1).contiguous()
#             vi = vi.permute(0, 2, 3, 1).contiguous()
#             # ir = ir.flatten(2).transpose(1, 2).contiguous()
#             # vi = vi.flatten(2).transpose(1, 2).contiguous()
#             ir = self.norms_ir[i](ir)
#             vi = self.norms_vi[i](vi)  # B, HW, C            
#             # ir = ir.reshape(B, h_ir, w_ir, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
#             # vi = vi.reshape(B, h_vi, w_vi, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
#             ir = ir.permute(0, 3, 1, 2).contiguous()
#             vi = vi.permute(0, 3, 1, 2).contiguous()
#             ir = self.conv_align_stage2_4[i](ir)
#             ir, vi = self.CSA_CMR_stages[i](ir, vi)            
#             fused_features_for_seg.append(self.cmaf_upsample[i](self.cmaf_stages[i](ir, vi)))
#             # 最后一个阶段的ir不需要再dealign了
#             if i != 2:
#                 ir = self.conv_dealign_stage2_4[i](ir)

#         # 记得最后要把segmentor输出 resize到和输入的图像一样大小
#         seg_out, hamburger = self.seg_decoder(fused_features_for_seg)
#         seg_out = F.interpolate(seg_out, size=(ori_h, ori_w), mode='bilinear', align_corners=False)

#         # fusion = self.eat_hamburger(fusion, hamburger)
#         fusion = self.fusion(fusion)

#         return fusion, seg_out

@ARCH_REGISTRY.register()
class RFEPFusion_no_register(nn.Module):
    def __init__(self, ir_settings = 'tiny', vi_settings = 'small', stem_scale = 2, 
                 vi_out_dim = 3,
                 ir_encoder_pretrained_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/ir_encoder.pth',
                 vi_encoder_pretrained_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/vi_encoder.pth'):
        super(RFEPFusion_no_register, self).__init__()
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
        
        vi_backbone = MSCAN(**self.vi_settings)
        ir_backbone = MSCAN(**self.ir_settings)
        
        self.vi_encoder = LLIE_Encoder(backbone_settings=self.vi_settings)
        self.ir_encoder = IrReconstructionEncoder(backbone_settings=self.ir_settings)
        
        if not os.path.exists(vi_encoder_pretrained_path):
            raise FileNotFoundError('LLIE_vi_encoder pretrained weights not found')
        if not os.path.exists(ir_encoder_pretrained_path):
            raise FileNotFoundError('LLIE_ir_encoder pretrained weights not found')        
        
        if stem_scale == 2:
            self.stem_ir = nn.Conv2d(in_channels=self.ir_settings['embed_dims'][0], 
                                     out_channels=self.ir_settings['embed_dims'][0], 
                                     kernel_size=3, 
                                     stride=2, 
                                     padding=1)
            self.stem_vi = nn.Conv2d(in_channels=self.vi_settings['embed_dims'][0], 
                                     out_channels=self.vi_settings['embed_dims'][0], 
                                     kernel_size=3, 
                                     stride=2, 
                                     padding=1)
        elif stem_scale == 4:
            self.stem_ir = StemConv(in_channels=self.ir_settings['embed_dims'][0], 
                                    out_channels=self.ir_settings['embed_dims'][0])
            self.stem_vi = StemConv(in_channels=self.vi_settings['embed_dims'][0], 
                                    out_channels=self.vi_settings['embed_dims'][0])

        self.blocks_ir = nn.ModuleList(
            [ir_backbone.block2, ir_backbone.block3, ir_backbone.block4]
        )
        self.norms_ir = nn.ModuleList(
            [ir_backbone.norm2, ir_backbone.norm3, ir_backbone.norm4]
        )
        self.patch_embeds_ir = nn.ModuleList(
            [ir_backbone.patch_embed2, ir_backbone.patch_embed3, ir_backbone.patch_embed4]
        )

        self.blocks_vi = nn.ModuleList(
            [vi_backbone.block2, vi_backbone.block3, vi_backbone.block4]
        )
        self.norms_vi = nn.ModuleList(
            [vi_backbone.norm2, vi_backbone.norm3, vi_backbone.norm4]
        )
        self.patch_embeds_vi = nn.ModuleList(
            [vi_backbone.patch_embed2, vi_backbone.patch_embed3, vi_backbone.patch_embed4]
        )

        if self.ir_settings['embed_dims'][0] != self.vi_settings['embed_dims'][0]:            
            self.conv_align_stage1 = nn.Conv2d(self.ir_settings['embed_dims'][0], self.vi_settings['embed_dims'][0], kernel_size=1, bias=True)
            self.conv_dealign_stage1 = nn.Conv2d(self.vi_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1, bias=True)
        else:
            self.conv_align_stage1 = nn.Identity()
            self.conv_dealign_stage1 = nn.Identity()

        self.conv_align_stage2_4 = nn.ModuleList()
        self.conv_dealign_stage2_4 = nn.ModuleList()
        for i in range(1, 4):
            if self.ir_settings['embed_dims'][i] != self.vi_settings['embed_dims'][i]:
                self.conv_align_stage2_4.append(nn.Conv2d(self.ir_settings['embed_dims'][i], self.vi_settings['embed_dims'][i], kernel_size=1, bias=True))
                if i != 3:
                    self.conv_dealign_stage2_4.append(nn.Conv2d(self.vi_settings['embed_dims'][i], self.ir_settings['embed_dims'][i], kernel_size=1, bias=True))
            else:
                self.conv_align_stage2_4.append(nn.Identity())
                if i != 3:
                    self.conv_dealign_stage2_4.append(nn.Identity())
               
        self.CSA_CMR_stage1_fusion = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][0])
        self.CSA_CMR_stages = nn.ModuleList(
            [CSA_CMR_Module(dim=self.vi_settings['embed_dims'][1]), 
             CSA_CMR_Module(dim=self.vi_settings['embed_dims'][2]), 
             CSA_CMR_Module(dim=self.vi_settings['embed_dims'][3])]
        )        

        self.cmaf_stage1_fusion = CMAF_Module(dim=self.vi_settings['embed_dims'][0], num_heads=1)
        self.cmaf_stages = nn.ModuleList(
            [CMAF_Module(dim=self.vi_settings['embed_dims'][1], num_heads=2), 
             CMAF_Module(dim=self.vi_settings['embed_dims'][2], num_heads=4), 
             CMAF_Module(dim=self.vi_settings['embed_dims'][3], num_heads=8)]
        )

        # self.cmaf_upsample = nn.ModuleList(
        #     [
        #         UpsampleConv(in_channels=self.vi_settings['embed_dims'][1], 
        #                  out_channels=self.vi_settings['embed_dims'][1],
        #                  scale = stem_scale,
        #                  norm_layer=self.vi_settings['norm_layer']), 
        #         UpsampleConv(in_channels=self.vi_settings['embed_dims'][2], 
        #                      out_channels=self.vi_settings['embed_dims'][2],
        #                      scale = stem_scale,
        #                      norm_layer=self.vi_settings['norm_layer']), 
        #         UpsampleConv(in_channels=self.vi_settings['embed_dims'][3], 
        #                      out_channels=self.vi_settings['embed_dims'][3],
        #                      scale = stem_scale,
        #                      norm_layer=self.vi_settings['norm_layer']),
        #     ] 
        # )
        self.cmaf_upsample = nn.ModuleList(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) for _ in range(3)
        )
        
        # self.eat_hamburger = SIM(norm_nc = self.vi_settings['embed_dims'][0], seg_nc = self.decode_head_settings['ham_channels'])
        self.fusion = MultiScaleFusion(in_channels=self.vi_settings['embed_dims'][0], mid_channels=64, out_channels=vi_out_dim, dilation_scales=[1, 2, 4])

        self.seg_decoder = MLPHead()
        
        # 这里直接用的params，没有考虑params_ema
        
        # 不再加载预训练模型的权重，不再冻结权重
        # self.vi_encoder.load_state_dict(torch.load(vi_encoder_pretrained_path)['params'], strict=True)        
        # self.ir_encoder.load_state_dict(torch.load(ir_encoder_pretrained_path)['params'], strict=True)
        # 冻结LLIE encoder的参数
        # print(list(self.vi_encoder.parameters())[0])
        # for param in self.vi_encoder.parameters():   
        # for param in list(self.vi_encoder.parameters())[:-2]:
        # for name, param in self.vi_encoder.named_parameters():
        #     # print(name)
        #     if not (name.startswith('block1.2')) or not (name.startswith('norm1')):
        #         param.requires_grad = False
        
        # for param in self.vi_encoder.parameters(): 
        #     param.requires_grad = False
        # # print(list(self.vi_encoder.parameters())[0])
        # for param in self.ir_encoder.parameters():
        #     param.requires_grad = False
            
        # for param in list(self.ir_encoder.parameters())[:-2]:
        # for name, param in self.ir_encoder.named_parameters():
        #     if not (name.startswith('block1.2')) or not (name.startswith('norm1')):
        #         param.requires_grad = False
    
    # 写一个计算过程和forward一样的函数，检测每个模块的输出（中间结果）是否正确（是否包含NAN值，打印张量的最大值和最小值）
    # 把GPU_num设为1，比较好定位错误
    @torch.no_grad()
    def forward_debug(self, ir, vi):        
        pass
        
    def forward(self, ir, vi):
        _, _, ori_h, ori_w = ir.shape
        # with torch.no_grad():
        vi = self.vi_encoder(vi)  # H, W
        ir = self.ir_encoder(ir)
        # print(ir)

        ir = self.conv_align_stage1(ir)
        ir, vi = self.CSA_CMR_stage1_fusion(ir, vi)
        fusion = self.cmaf_stage1_fusion(ir, vi)
        # print(ir)

        ir = self.conv_dealign_stage1(ir)
        ir = self.stem_ir(ir)   # H/2, W/2
        vi = self.stem_vi(vi)

        fused_features_for_seg = []  # [(H/2, W/2), (H/4, W/4), (H/8, W/8)]

        # B = ir.shape[0]

        # stage1: H/4, W/4
        # stage2: H/8, W/8
        # stage3: H/16, W/16
        for i in range(3):
            ir, h_ir, w_ir = self.patch_embeds_ir[i](ir)  
            vi, h_vi, w_vi = self.patch_embeds_vi[i](vi)  
            # block = self.blocks_ir[i]
            # ir = block(ir, h_ir, w_ir)
            # ir = self.blocks_ir[i](ir, h_ir, w_ir)
            # vi = self.blocks_vi[i](vi, h_vi, w_vi)
            for blk in self.blocks_ir[i]:
                ir = blk(ir, h_ir, w_ir)
            for blk in self.blocks_vi[i]:             
                vi = blk(vi, h_vi, w_vi)
            ir = ir.permute(0, 2, 3, 1).contiguous()
            vi = vi.permute(0, 2, 3, 1).contiguous()
            # ir = ir.flatten(2).transpose(1, 2).contiguous()
            # vi = vi.flatten(2).transpose(1, 2).contiguous()
            ir = self.norms_ir[i](ir)
            vi = self.norms_vi[i](vi)  # B, HW, C            
            # ir = ir.reshape(B, h_ir, w_ir, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            # vi = vi.reshape(B, h_vi, w_vi, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            ir = ir.permute(0, 3, 1, 2).contiguous()
            vi = vi.permute(0, 3, 1, 2).contiguous()
            ir = self.conv_align_stage2_4[i](ir)
            ir, vi = self.CSA_CMR_stages[i](ir, vi)            
            fused_features_for_seg.append(self.cmaf_upsample[i](self.cmaf_stages[i](ir, vi)))
            # 最后一个阶段的ir不需要再dealign了
            if i != 2:
                ir = self.conv_dealign_stage2_4[i](ir)

        # 记得最后要把segmentor输出 resize到和输入的图像一样大小
        # seg_out, hamburger = self.seg_decoder(fused_features_for_seg)
        seg_out = self.seg_decoder(fused_features_for_seg)
        seg_out = F.interpolate(seg_out, size=(ori_h, ori_w), mode='bilinear', align_corners=False)

        # fusion = self.eat_hamburger(fusion, hamburger)
        fusion = self.fusion(fusion)

        return fusion, seg_out


@ARCH_REGISTRY.register()
class RFEPFusion_no_register_UPerNet(nn.Module):
    def __init__(self, ir_settings = 'tiny', vi_settings = 'small', stem_scale = 2, 
                 vi_out_dim = 3,
                 ir_encoder_pretrained_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/ir_encoder.pth',
                 vi_encoder_pretrained_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/vi_encoder.pth'):
        super(RFEPFusion_no_register_UPerNet, self).__init__()
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
        
        vi_backbone = MSCAN(**self.vi_settings)
        ir_backbone = MSCAN(**self.ir_settings)
        
        self.vi_encoder = LLIE_Encoder(backbone_settings=self.vi_settings)
        self.ir_encoder = IrReconstructionEncoder(backbone_settings=self.ir_settings)
        
        if not os.path.exists(vi_encoder_pretrained_path):
            raise FileNotFoundError('LLIE_vi_encoder pretrained weights not found')
        if not os.path.exists(ir_encoder_pretrained_path):
            raise FileNotFoundError('LLIE_ir_encoder pretrained weights not found')        
        
        if stem_scale == 2:
            self.stem_ir = nn.Conv2d(in_channels=self.ir_settings['embed_dims'][0], 
                                     out_channels=self.ir_settings['embed_dims'][0], 
                                     kernel_size=3, 
                                     stride=2, 
                                     padding=1)
            self.stem_vi = nn.Conv2d(in_channels=self.vi_settings['embed_dims'][0], 
                                     out_channels=self.vi_settings['embed_dims'][0], 
                                     kernel_size=3, 
                                     stride=2, 
                                     padding=1)
        elif stem_scale == 4:
            self.stem_ir = StemConv(in_channels=self.ir_settings['embed_dims'][0], 
                                    out_channels=self.ir_settings['embed_dims'][0])
            self.stem_vi = StemConv(in_channels=self.vi_settings['embed_dims'][0], 
                                    out_channels=self.vi_settings['embed_dims'][0])

        self.blocks_ir = nn.ModuleList(
            [ir_backbone.block2, ir_backbone.block3, ir_backbone.block4]
        )
        self.norms_ir = nn.ModuleList(
            [ir_backbone.norm2, ir_backbone.norm3, ir_backbone.norm4]
        )
        self.patch_embeds_ir = nn.ModuleList(
            [ir_backbone.patch_embed2, ir_backbone.patch_embed3, ir_backbone.patch_embed4]
        )

        self.blocks_vi = nn.ModuleList(
            [vi_backbone.block2, vi_backbone.block3, vi_backbone.block4]
        )
        self.norms_vi = nn.ModuleList(
            [vi_backbone.norm2, vi_backbone.norm3, vi_backbone.norm4]
        )
        self.patch_embeds_vi = nn.ModuleList(
            [vi_backbone.patch_embed2, vi_backbone.patch_embed3, vi_backbone.patch_embed4]
        )

        if self.ir_settings['embed_dims'][0] != self.vi_settings['embed_dims'][0]:            
            self.conv_align_stage1 = nn.Conv2d(self.ir_settings['embed_dims'][0], self.vi_settings['embed_dims'][0], kernel_size=1, bias=True)
            self.conv_dealign_stage1 = nn.Conv2d(self.vi_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1, bias=True)
        else:
            self.conv_align_stage1 = nn.Identity()
            self.conv_dealign_stage1 = nn.Identity()

        self.conv_align_stage2_4 = nn.ModuleList()
        self.conv_dealign_stage2_4 = nn.ModuleList()
        for i in range(1, 4):
            if self.ir_settings['embed_dims'][i] != self.vi_settings['embed_dims'][i]:
                self.conv_align_stage2_4.append(nn.Conv2d(self.ir_settings['embed_dims'][i], self.vi_settings['embed_dims'][i], kernel_size=1, bias=True))
                if i != 3:
                    self.conv_dealign_stage2_4.append(nn.Conv2d(self.vi_settings['embed_dims'][i], self.ir_settings['embed_dims'][i], kernel_size=1, bias=True))
            else:
                self.conv_align_stage2_4.append(nn.Identity())
                if i != 3:
                    self.conv_dealign_stage2_4.append(nn.Identity())
               
        self.CSA_CMR_stage1_fusion = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][0])
        self.CSA_CMR_stages = nn.ModuleList(
            [CSA_CMR_Module(dim=self.vi_settings['embed_dims'][1]), 
             CSA_CMR_Module(dim=self.vi_settings['embed_dims'][2]), 
             CSA_CMR_Module(dim=self.vi_settings['embed_dims'][3])]
        )        

        self.cmaf_stage1_fusion = CMAF_Module(dim=self.vi_settings['embed_dims'][0], num_heads=1)
        self.cmaf_stages = nn.ModuleList(
            [CMAF_Module(dim=self.vi_settings['embed_dims'][1], num_heads=2), 
             CMAF_Module(dim=self.vi_settings['embed_dims'][2], num_heads=4), 
             CMAF_Module(dim=self.vi_settings['embed_dims'][3], num_heads=8)]
        )
        
        self.cmaf_upsample = nn.ModuleList(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) for _ in range(3)
        )
        
        self.fusion = MultiScaleFusion(in_channels=self.vi_settings['embed_dims'][0], mid_channels=64, out_channels=vi_out_dim, dilation_scales=[1, 2, 4])

        self.seg_decoder = UPerHead(in_channels=self.vi_settings['embed_dims'], num_classes=9,
                                    channels=512, pool_scales=(1,2,3,6))
        
        # 这里直接用的params，没有考虑params_ema
        
        self.vi_encoder.load_state_dict(torch.load(vi_encoder_pretrained_path)['params'], strict=True)        
        self.ir_encoder.load_state_dict(torch.load(ir_encoder_pretrained_path)['params'], strict=True)
        
        for param in self.vi_encoder.parameters(): 
            param.requires_grad = False
        # print(list(self.vi_encoder.parameters())[0])
        for param in self.ir_encoder.parameters():
            param.requires_grad = False
        
    def forward(self, ir, vi):
        _, _, ori_h, ori_w = ir.shape
        # with torch.no_grad():
        vi = self.vi_encoder(vi)  # H, W
        ir = self.ir_encoder(ir)
        # print(ir)

        ir = self.conv_align_stage1(ir)
        ir, vi = self.CSA_CMR_stage1_fusion(ir, vi)
        fusion = self.cmaf_stage1_fusion(ir, vi)
        seg_fusion = fusion.clone()
        # print(ir)

        ir = self.conv_dealign_stage1(ir)
        ir = self.stem_ir(ir)   # H/2, W/2
        vi = self.stem_vi(vi)

        fused_features_for_seg = [seg_fusion]  # [(H/2, W/2), (H/4, W/4), (H/8, W/8)]

        # B = ir.shape[0]

        # stage1: H/4, W/4
        # stage2: H/8, W/8
        # stage3: H/16, W/16
        for i in range(3):
            ir, h_ir, w_ir = self.patch_embeds_ir[i](ir)  
            vi, h_vi, w_vi = self.patch_embeds_vi[i](vi)  
            # block = self.blocks_ir[i]
            # ir = block(ir, h_ir, w_ir)
            # ir = self.blocks_ir[i](ir, h_ir, w_ir)
            # vi = self.blocks_vi[i](vi, h_vi, w_vi)
            for blk in self.blocks_ir[i]:
                ir = blk(ir, h_ir, w_ir)
            for blk in self.blocks_vi[i]:             
                vi = blk(vi, h_vi, w_vi)
            ir = ir.permute(0, 2, 3, 1).contiguous()
            vi = vi.permute(0, 2, 3, 1).contiguous()
            # ir = ir.flatten(2).transpose(1, 2).contiguous()
            # vi = vi.flatten(2).transpose(1, 2).contiguous()
            ir = self.norms_ir[i](ir)
            vi = self.norms_vi[i](vi)  # B, HW, C            
            # ir = ir.reshape(B, h_ir, w_ir, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            # vi = vi.reshape(B, h_vi, w_vi, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            ir = ir.permute(0, 3, 1, 2).contiguous()
            vi = vi.permute(0, 3, 1, 2).contiguous()
            ir = self.conv_align_stage2_4[i](ir)
            ir, vi = self.CSA_CMR_stages[i](ir, vi) 
            # fused_features_for_seg.append(self.cmaf_stages[i](ir, vi))
            fused_features_for_seg.append(self.cmaf_upsample[i](self.cmaf_stages[i](ir, vi)))
            # 最后一个阶段的ir不需要再dealign了
            if i != 2:
                ir = self.conv_dealign_stage2_4[i](ir)

        # 记得最后要把segmentor输出 resize到和输入的图像一样大小
        # seg_out, hamburger = self.seg_decoder(fused_features_for_seg)
        fused_features_for_seg[0] = F.interpolate(fused_features_for_seg[0], 
                                                  size=(fused_features_for_seg[1].shape[2:]),
                                                  mode='bilinear')
        seg_out = self.seg_decoder(fused_features_for_seg)  # (H/4, W/4)
        seg_out = F.interpolate(seg_out, size=(ori_h, ori_w), mode='bilinear', align_corners=False)

        # fusion = self.eat_hamburger(fusion, hamburger)
        fusion = self.fusion(fusion)

        return fusion, seg_out

# @ARCH_REGISTRY.register()
# class RFEPFusion(RFEPFusion_no_register):
#     def __init__(self):
#         super(RFEPFusion, self).__init__()
        
#     def forward(self, ir, vi):
#         pass
