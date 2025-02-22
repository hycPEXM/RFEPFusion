import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .mscan import StemConv, tiny_settings, small_settings, MSCAN
from .LLIE_arch import LLIE_Encoder, IrReconstructionEncoder
from .fusion_modules import CSA_CMR_Module, CMAF_Module, SIM, MultiScaleFusion
from .seg_decoder import LightHamHead, small_settings_decode_head
from .net_utils import UpsampleConv

# Asymmetric dual-stream/dual-path backbone (using backbone variant of different sizes for different modalities)
# symmetric fusion modules
@ARCH_REGISTRY.register()
class RFEPFusion_no_register(nn.Module):
    def __init__(self, ir_settings = tiny_settings, vi_settings = small_settings, stem_scale = 2):
        super(RFEPFusion_no_register, self).__init__()
        vi_backbone = MSCAN(**vi_settings)
        ir_backbone = MSCAN(**ir_settings)
        self.vi_encoder = LLIE_Encoder()
        self.ir_encoder = IrReconstructionEncoder()
        if stem_scale == 2:
            self.stem_ir = nn.Conv2d(in_channels=ir_settings['embed_dims'][0], 
                                     out_channels=ir_settings['embed_dims'][0], 
                                     kernel_size=3, 
                                     stride=2, 
                                     padding=1)
            self.stem_vi = nn.Conv2d(in_channels=vi_settings['embed_dims'][0], 
                                     out_channels=vi_settings['embed_dims'][0], 
                                     kernel_size=3, 
                                     stride=2, 
                                     padding=1)
        elif stem_scale == 4:
            self.stem_ir = StemConv(in_channels=ir_settings['embed_dims'][0], 
                                    out_channels=ir_settings['embed_dims'][0])
            self.stem_vi = StemConv(in_channels=vi_settings['embed_dims'][0], 
                                    out_channels=vi_settings['embed_dims'][0])

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

        self.conv_align_stage1 = nn.Conv2d(ir_settings['embed_dims'][0], vi_settings['embed_dims'][0], kernel_size=1, bias=True)
        self.conv_dealign_stage1 = nn.Conv2d(vi_settings['embed_dims'][0], ir_settings['embed_dims'][0], kernel_size=1, bias=True)
        self.conv_align_stage2 = nn.Conv2d(ir_settings['embed_dims'][1], vi_settings['embed_dims'][1], kernel_size=1, bias=True)
        self.conv_dealign_stage2 = nn.Conv2d(vi_settings['embed_dims'][1], ir_settings['embed_dims'][1], kernel_size=1, bias=True)
        self.conv_align_stage3 = nn.Conv2d(ir_settings['embed_dims'][2], vi_settings['embed_dims'][2], kernel_size=1, bias=True)
        self.conv_dealign_stage3 = nn.Conv2d(vi_settings['embed_dims'][2], ir_settings['embed_dims'][2], kernel_size=1, bias=True)
        self.conv_align_stage4 = nn.Conv2d(ir_settings['embed_dims'][3], vi_settings['embed_dims'][3], kernel_size=1, bias=True)
        self.conv_dealign_stage4 = nn.Conv2d(vi_settings['embed_dims'][3], ir_settings['embed_dims'][3], kernel_size=1, bias=True)

        self.CSA_CMR_stage1_fusion = CSA_CMR_Module(dim=vi_settings['embed_dims'][0])
        self.CSA_CMR_stages = nn.ModuleList(
            [CSA_CMR_Module(dim=vi_settings['embed_dims'][1]), 
             CSA_CMR_Module(dim=vi_settings['embed_dims'][2]), 
             CSA_CMR_Module(dim=vi_settings['embed_dims'][3])]
        )
        # self.CSA_CMR_stage2 = CSA_CMR_Module(dim=vi_settings['embed_dims'][1])
        # self.CSA_CMR_stage3 = CSA_CMR_Module(dim=vi_settings['embed_dims'][2])
        # self.CSA_CMR_stage4 = CSA_CMR_Module(dim=vi_settings['embed_dims'][3])

        self.cmaf_stage1_fusion = CMAF_Module(dim=vi_settings['embed_dims'][0])
        self.cmaf_stages = nn.ModuleList(
            [CMAF_Module(dim=vi_settings['embed_dims'][1]), 
             CMAF_Module(dim=vi_settings['embed_dims'][2]), 
             CMAF_Module(dim=vi_settings['embed_dims'][3])]
        )
        # self.cmaf_stage2 = CMAF_Module(dim=vi_settings['embed_dims'][1])
        # self.cmaf_stage3 = CMAF_Module(dim=vi_settings['embed_dims'][2])
        # self.cmaf_stage4 = CMAF_Module(dim=vi_settings['embed_dims'][3])

        self.cmaf_upsample = nn.ModuleList(
            [
                UpsampleConv(in_channels=vi_settings['embed_dims'][1], 
                         out_channels=vi_settings['embed_dims'][1],
                         scale = stem_scale,
                         norm_layer=vi_settings['norm_layer']), 
                UpsampleConv(in_channels=vi_settings['embed_dims'][2], 
                             out_channels=vi_settings['embed_dims'][2],
                             scale = stem_scale,
                             norm_layer=vi_settings['norm_layer']), 
                UpsampleConv(in_channels=vi_settings['embed_dims'][3], 
                             out_channels=vi_settings['embed_dims'][3],
                             scale = stem_scale,
                             norm_layer=vi_settings['norm_layer']),
            ] 
        )
        
        self.SIM = SIM(norm_nc = vi_settings['embed_dims'][0], seg_nc = small_settings_decode_head['ham_channels'])
        self.fusion = MultiScaleFusion(in_channels=vi_settings['embed_dims'][0], mid_channels=32, out_channels=3, dilation_scales=[1, 2, 4])

        self.seg_decoder = LightHamHead(**small_settings_decode_head)

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
        B = ir.shape[0]
        # stage1: H/4, W/4
        # stage2: H/8, W/8
        # stage3: H/16, W/16
        for i in range(3):
            ir, h_ir, w_ir = self.patch_embeds_ir[i](ir)  
            vi, h_vi, w_vi = self.patch_embeds_vi[i](vi)  
            ir = self.blocks_ir[i](ir, h_ir, w_ir)
            vi = self.blocks_vi[i](vi, h_vi, w_vi)
            ir = ir.flatten(2).transpose(1, 2).contiguous()
            vi = vi.flatten(2).transpose(1, 2).contiguous()
            ir = self.norms_ir[i](ir)
            vi = self.norms_vi[i](vi)  # B, HW, C            
            ir = ir.reshape(B, h_ir, w_ir, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            vi = vi.reshape(B, h_vi, w_vi, -1).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
            ir, vi = self.CSA_CMR_stages[i](ir, vi)
            fused_features_for_seg.append(self.cmaf_upsample[i](self.cmaf_stages[i](ir, vi)))

        # 记得最后要把segmentor输出 resize到和输入的图像一样大小
        seg_out, ham = self.seg_decoder(fused_features_for_seg)
        seg_out = F.interpolate(seg_out, size=(ori_h, ori_w), mode='bilinear', align_corners=False)

        fusion = self.SIM(fusion, ham)
        fusion = self.fusion(fusion)

        return fusion, seg_out


@ARCH_REGISTRY.register()
class RFEPFusion(RFEPFusion_no_register):
    def __init__(self):
        super(RFEPFusion, self).__init__()
        
    def forward(self, ir, vi):
        pass
