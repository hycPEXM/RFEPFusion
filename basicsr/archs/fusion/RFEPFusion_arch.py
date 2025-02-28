import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

import os

from .mscan import StemConv, tiny_settings, small_settings, base_settings, hyc_small_settings, MSCAN
from .LLIE_arch import LLIE_Encoder, IrReconstructionEncoder
from .fusion_modules import CSA_CMR_Module, CMAF_Module, SIM, MultiScaleFusion
from .seg_decoder import *
from .net_utils import UpsampleConv

# Asymmetric dual-stream/dual-path backbone (using backbone variant of different sizes for different modalities)
# symmetric fusion modules
@ARCH_REGISTRY.register()
class RFEPFusion_no_register(nn.Module):
    def __init__(self, ir_settings = 'tiny', vi_settings = 'small', stem_scale = 2, 
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
        
        if not os.path.exists(vi_encoder_pretrained_path):
            raise FileNotFoundError('LLIE_vi_encoder pretrained weights not found')
        if not os.path.exists(ir_encoder_pretrained_path):
            raise FileNotFoundError('LLIE_ir_encoder pretrained weights not found')
        self.vi_encoder = LLIE_Encoder(backbone_settings=self.vi_settings).load_state_dict(torch.load(vi_encoder_pretrained_path), strict=True)
        self.ir_encoder = IrReconstructionEncoder(backbone_settings=self.ir_settings).load_state_dict(torch.load(ir_encoder_pretrained_path), strict=True)
        # 冻结LLIE encoder的参数
        for param in self.vi_encoder.parameters():
            param.requires_grad = False
        for param in self.ir_encoder.parameters():
            param.requires_grad = False
        
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

        # self.block2_vi = vi_backbone.block2
        # self.norm2_vi = vi_backbone.norm2
        # self.patch_embed2_vi = vi_backbone.patch_embed2
        # self.block3_vi = vi_backbone.block3
        # self.norm3_vi = vi_backbone.norm3
        # self.patch_embed3_vi = vi_backbone.patch_embed3
        # self.block4_vi = vi_backbone.block4
        # self.norm4_vi = vi_backbone.norm4
        # self.patch_embed4_vi = vi_backbone.patch_embed4

        # self.block2_ir = ir_backbone.block2
        # self.norm2_ir = ir_backbone.norm2
        # self.patch_embed2_ir = ir_backbone.patch_embed2
        # self.block3_ir = ir_backbone.block3
        # self.norm3_ir = ir_backbone.norm3
        # self.patch_embed3_ir = ir_backbone.patch_embed3
        # self.block4_ir = ir_backbone.block4
        # self.norm4_ir = ir_backbone.norm4
        # self.patch_embed4_ir = ir_backbone.patch_embed4

        if self.ir_settings['embed_dims'][0] != self.vi_settings['embed_dims'][0]:
            self.conv_align_stage1 = nn.Conv2d(self.ir_settings['embed_dims'][0], self.vi_settings['embed_dims'][0], kernel_size=1, bias=True)
            self.conv_dealign_stage1 = nn.Conv2d(self.vi_settings['embed_dims'][0], self.ir_settings['embed_dims'][0], kernel_size=1, bias=True)
        else:
            self.conv_align_stage1 = nn.Identity()
            self.conv_dealign_stage1 = nn.Identity()


        self.conv_align_stage2_4 = nn.ModuleList()
        self.conv_dealign_stage2_4 = nn.ModuleList()
        if self.ir_settings['embed_dims'][1] != self.vi_settings['embed_dims'][1]:
            self.conv_align_stage2_4.append(nn.Conv2d(self.ir_settings['embed_dims'][1], self.vi_settings['embed_dims'][1], kernel_size=1, bias=True))
            self.conv_dealign_stage2_4.append(nn.Conv2d(self.vi_settings['embed_dims'][1], self.ir_settings['embed_dims'][1], kernel_size=1, bias=True))
        else:
            self.conv_align_stage2_4.append(nn.Identity())
            self.conv_dealign_stage2_4.append(nn.Identity())
        if self.ir_settings['embed_dims'][2] != self.vi_settings['embed_dims'][2]:
            self.conv_align_stage2_4.append(nn.Conv2d(self.ir_settings['embed_dims'][2], self.vi_settings['embed_dims'][2], kernel_size=1, bias=True))
            self.conv_dealign_stage2_4.append(nn.Conv2d(self.vi_settings['embed_dims'][2], self.ir_settings['embed_dims'][2], kernel_size=1, bias=True))
        else:
            self.conv_align_stage2_4.append(nn.Identity())
            self.conv_dealign_stage2_4.append(nn.Identity())
        if self.ir_settings['embed_dims'][3] != self.vi_settings['embed_dims'][3]:
            self.conv_align_stage2_4.append(nn.Conv2d(self.ir_settings['embed_dims'][3], self.vi_settings['embed_dims'][3], kernel_size=1, bias=True))
            self.conv_dealign_stage2_4.append(nn.Conv2d(self.vi_settings['embed_dims'][3], self.ir_settings['embed_dims'][3], kernel_size=1, bias=True))
        else:
            self.conv_align_stage2_4.append(nn.Identity())
            self.conv_dealign_stage2_4.append(nn.Identity())        

        self.CSA_CMR_stage1_fusion = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][0])
        self.CSA_CMR_stages = nn.ModuleList(
            [CSA_CMR_Module(dim=self.vi_settings['embed_dims'][1]), 
             CSA_CMR_Module(dim=self.vi_settings['embed_dims'][2]), 
             CSA_CMR_Module(dim=self.vi_settings['embed_dims'][3])]
        )
        # self.CSA_CMR_stage2 = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][1])
        # self.CSA_CMR_stage3 = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][2])
        # self.CSA_CMR_stage4 = CSA_CMR_Module(dim=self.vi_settings['embed_dims'][3])

        self.cmaf_stage1_fusion = CMAF_Module(dim=self.vi_settings['embed_dims'][0])
        self.cmaf_stages = nn.ModuleList(
            [CMAF_Module(dim=self.vi_settings['embed_dims'][1]), 
             CMAF_Module(dim=self.vi_settings['embed_dims'][2]), 
             CMAF_Module(dim=self.vi_settings['embed_dims'][3])]
        )
        # self.cmaf_stage2 = CMAF_Module(dim=self.vi_settings['embed_dims'][1])
        # self.cmaf_stage3 = CMAF_Module(dim=self.vi_settings['embed_dims'][2])
        # self.cmaf_stage4 = CMAF_Module(dim=self.vi_settings['embed_dims'][3])

        self.cmaf_upsample = nn.ModuleList(
            [
                UpsampleConv(in_channels=self.vi_settings['embed_dims'][1], 
                         out_channels=self.vi_settings['embed_dims'][1],
                         scale = stem_scale,
                         norm_layer=self.vi_settings['norm_layer']), 
                UpsampleConv(in_channels=self.vi_settings['embed_dims'][2], 
                             out_channels=self.vi_settings['embed_dims'][2],
                             scale = stem_scale,
                             norm_layer=self.vi_settings['norm_layer']), 
                UpsampleConv(in_channels=self.vi_settings['embed_dims'][3], 
                             out_channels=self.vi_settings['embed_dims'][3],
                             scale = stem_scale,
                             norm_layer=self.vi_settings['norm_layer']),
            ] 
        )

        if vi_settings == 'small':
            self.decode_head_settings = small_settings_decode_head 
        elif vi_settings == 'tiny':
            self.decode_head_settings = tiny_settings_decode_head
        elif vi_settings == 'base':
            self.decode_head_settings = base_settings_decode_head
        elif vi_settings == 'hyc_small':
            self.decode_head_settings = small_settings_decode_head 
        
        self.eat_hamburger = SIM(norm_nc = self.vi_settings['embed_dims'][0], seg_nc = self.decode_head_settings['ham_channels'])
        self.fusion = MultiScaleFusion(in_channels=self.vi_settings['embed_dims'][0], mid_channels=32, out_channels=3, dilation_scales=[1, 2, 4])

        self.seg_decoder = LightHamHead(**self.decode_head_settings)

    def forward(self, ir, vi):
        _, _, ori_h, ori_w = ir.shape
        vi = self.vi_encoder(vi)  # H, W
        ir = self.ir_encoder(ir)

        ir = self.conv_align_stage1(ir)
        ir, vi = self.CSA_CMR_stage1_fusion(ir, vi)
        fusion = self.cmaf_stage1_fusion(ir, vi)

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
            ir = self.blocks_ir[i](ir, h_ir, w_ir)
            vi = self.blocks_vi[i](vi, h_vi, w_vi)
            ir = ir.permute(0, 2, 3, 1).contiguous()
            vi = vi.permute(0, 2, 3, 1).contiguous()
            # ir = ir.flatten(2).transpose(1, 2).contiguous()
            # vi = vi.flatten(2).transpose(1, 2).contiguous()
            ir = self.norms_ir[i](ir)
            vi = self.norms_vi[i](vi)  # B, HW, C            
            # ir = ir.reshape(B, h_ir, w_ir, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            # vi = vi.reshape(B, h_vi, w_vi, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            ir = self.conv_align_stage2_4[i](ir)
            ir, vi = self.CSA_CMR_stages[i](ir, vi)            
            fused_features_for_seg.append(self.cmaf_upsample[i](self.cmaf_stages[i](ir, vi)))
            ir = self.conv_dealign_stage2_4[i](ir)

        # 记得最后要把segmentor输出 resize到和输入的图像一样大小
        seg_out, hamburger = self.seg_decoder(fused_features_for_seg)
        seg_out = F.interpolate(seg_out, size=(ori_h, ori_w), mode='bilinear', align_corners=False)

        fusion = self.eat_hamburger(fusion, hamburger)
        fusion = self.fusion(fusion)

        return fusion, seg_out


@ARCH_REGISTRY.register()
class RFEPFusion(RFEPFusion_no_register):
    def __init__(self):
        super(RFEPFusion, self).__init__()
        
    def forward(self, ir, vi):
        pass
