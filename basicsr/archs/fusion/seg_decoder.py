import torch
import torch.nn as nn
import torch.nn.functional as F

# from mmcv.cnn import ConvModule
# from mmseg.ops import resize
# from ..builder import HEADS
# from .decode_head import BaseDecodeHead

# __all__ = ['LightHamHead', 'base_settings_decode_head', 'small_settings_decode_head', 'tiny_settings_decode_head']

class ConvModule(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1,
                 padding=0,
                 norm_cfg=None,
                 act_cfg=True):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False if norm_cfg else True
        )
        # 'Unnecessary conv bias before batch/instance norm'
        
        # 处理标准化层
        if norm_cfg is None:
            self.norm = None
        else:
            if norm_cfg['type'] == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_cfg['type'] == 'GN':
                self.norm = nn.GroupNorm(
                    num_groups=norm_cfg.get('num_groups', 32),
                    num_channels=out_channels
                )
            elif norm_cfg['type'] == 'LN':
                self.norm = nn.LayerNorm(out_channels)
            elif norm_cfg['type'] == 'IN':
                self.norm = nn.InstanceNorm2d(out_channels)
            
            # 设置是否需要梯度
            for param in self.norm.parameters():
                param.requires_grad = norm_cfg.get('requires_grad', True)
        
        self.act = nn.GELU() if act_cfg else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        # self.eval_steps = args.setdefault('EVAL_STEPS', 7)
        self.eval_steps = args.setdefault('EVAL_STEPS', 6)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

        # print('spatial', self.spatial)
        # print('S', self.S)
        # print('D', self.D)
        # print('R', self.R)
        # print('train_steps', self.train_steps)
        # print('eval_steps', self.eval_steps)
        # print('inv_t', self.inv_t)
        # print('eta', self.eta)
        # print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2).contiguous(), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2).contiguous()

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2).contiguous())

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        # bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2).contiguous(), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).contiguous().bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)      # coef * (x^T * bases) / (coef * bases^T * bases)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).contiguous().bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)    # bases * (x * coef) / (bases * coef^T * coef)
        # bases: (B * S, D, R); coef: (B * S, N, R)
        return bases, coef
    
    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2).contiguous(), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).contiguous().bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=None,
            act_cfg=None
        )

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        # enjoy = F.relu(enjoy, inplace=True)
        enjoy = F.gelu(enjoy)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        # ham = F.relu(x + enjoy, inplace=True)

        # return ham
        return F.gelu(x + enjoy)


class LightHamHead(nn.Module):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO: 
        Add other MD models (Ham). 
    """

    def __init__(self,
                in_channels,
                channels,
                *,
                num_classes,
                dropout_ratio=0.1,
                # conv_cfg=None,
                norm_cfg=None,
                act_cfg=True,
                in_index=-1,
                input_transform=None,                
                ignore_index=255,
                # sampler=None,
                align_corners=False,
                ham_channels=512,
                ham_kwargs=dict(),
                ):
        super(LightHamHead, self).__init__()
        
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        # self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        # if isinstance(loss_decode, dict):
        #     self.loss_decode = build_loss(loss_decode)
        # elif isinstance(loss_decode, (list, tuple)):
        #     self.loss_decode = nn.ModuleList()
        #     for loss in loss_decode:
        #         self.loss_decode.append(build_loss(loss))
        # else:
        #     raise TypeError(f'loss_decode must be a dict or sequence of dict,\
        #         but got {type(loss_decode)}')

        # if sampler is not None:
        #     self.sampler = build_pixel_sampler(sampler, context=self)
        # else:
        #     self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        # self.fp16_enabled = False

        self.ham_channels = ham_channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.ham_channels,
            1,
            # conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.hamburger = Hamburger(ham_channels, ham_kwargs, norm_cfg=self.norm_cfg)

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            # conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels
        
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        # redundant/unnecessary computation because self.in_index is [0, 1, 2]
        # inputs = self._transform_inputs(inputs) 

        inputs = [inputs[0]] + [F.interpolate(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs[1:]]

        inputs = torch.cat(inputs, dim=1)
        # print(inputs.shape)
        x = self.squeeze(inputs)
        # del inputs
        x = self.hamburger(x)  # B,C=(256),H,W

        output = self.align(x)
        output = self.cls_seg(output)
        return output, x


base_settings_decode_head = dict(
    in_channels=[128, 320, 512],
    in_index=[0, 1, 2],
    channels=512,
    ham_channels=512,
    ham_kwargs=dict(MD_R=64),
    dropout_ratio=0.1,
    num_classes=9,
    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    align_corners=False,
    input_transform='multiple_select'
)

small_settings_decode_head = dict(
    in_channels=[128, 320, 512],
    in_index=[0, 1, 2],
    channels=256,
    ham_channels=256,
    ham_kwargs=dict(MD_R=16),
    dropout_ratio=0.1,
    num_classes=9,
    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    align_corners=False,
    input_transform='multiple_select'
    )

tiny_settings_decode_head = dict(
    in_channels=[64, 160, 256],
    in_index=[0, 1, 2],
    channels=256,
    ham_channels=256,
    ham_kwargs=dict(MD_R=16),
    dropout_ratio=0.1,
    num_classes=9,
    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
    align_corners=False,
    input_transform='multiple_select'
)

# import torch.nn as nn
# import torch
# import torch.nn.functional as F

class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.
    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels=[96, 192, 384, 768], num_classes=40, channels=512, pool_scales=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d, dropout_ratio=0.1, align_corners=False):
        super(UPerHead, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            norm_layer=norm_layer,
            align_corners=align_corners)
        self.bottleneck = nn.Sequential(
                nn.Conv2d(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, 3, padding=1,bias=False),
                norm_layer(self.channels),
                # nn.ReLU(inplace=True)
                nn.GELU()
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.channels, 1, bias=False),
                norm_layer(self.channels),
                # nn.ReLU()
                nn.GELU()
                )
            fpn_conv = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
                norm_layer(self.channels),
                # nn.ReLU()
                nn.GELU()
                )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = nn.Sequential(
                nn.Conv2d(len(self.in_channels) * self.channels, self.channels, 3, padding=1, bias=False),
                norm_layer(self.channels),
                # nn.ReLU(inplace=True)
                nn.GELU()
                )
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.conv_seg(output)

        return output


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channel, channels, norm_layer, align_corners=False):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channel = in_channel
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channel, self.channels, 1, bias=False),
                    norm_layer(self.channels), 
                    # nn.ReLU(inplace=True)
                    nn.GELU()
                ))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs
    