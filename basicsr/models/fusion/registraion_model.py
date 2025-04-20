import torch
from collections import OrderedDict
from os import path as osp
import os
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

import torchvision
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, scandir
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel

from basicsr.data.fusion_utils import RGB2YCrCb, YCbCr2RGB

# from .auto_weighting import *
import basicsr.models.fusion.auto_weighting as weighting

# from PIL import Image

import datetime

try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False

import random
import cv2
import numpy as np

@MODEL_REGISTRY.register()
class RFEPFusionRegistrationModel(BaseModel):
    def __init__(self, opt):
        super(RFEPFusionRegistrationModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)                
        
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if opt['dist'] or opt['num_gpu']>1:
            print("Turning BatchNorm to SyncBN!")
            self.net_g = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_g)
        
        # self.num_task = 2 # fusion + segmentation
        
        # self.weighting_strategy = None
        # if self.is_train:
        #     self.weighting_strategy = opt['train'].get('MTL_auto_weighting')
                        
        # ‌Automatic Mixed Precision Training
        self.use_amp = opt.get('use_amp', False) and load_amp        
        if self.use_amp:
            self.amp_scaler = GradScaler(enabled=self.use_amp)
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')     
            
        self.dynamic_shape = self.opt.get('dynamic_shape', False)
        if not self.dynamic_shape:
            if isinstance(self.opt['datasets']['train']['gt_size'], list):
                assert len(self.opt['datasets']['train']['gt_size']) == 2, "gt_size must be a list of two integers: [width, height]"
                self.patch_w, self.patch_h = self.opt['datasets']['train']['gt_size']
            elif isinstance(self.opt['datasets']['train']['gt_size'], int):
                self.patch_w = self.patch_h = self.opt['datasets']['train']['gt_size']
            import kornia.utils as KU
            self.grid = KU.create_meshgrid(self.patch_h, self.patch_w)
            self.grid = self.grid.type(torch.FloatTensor).to(self.device)           
        
        # 目前支持amp与weighting一起使用，但不推荐
        # if self.weighting_strategy and self.use_amp:
        #     raise ValueError('auto weighting currently can\'t used with amp training') 
        
        # if self.weighting_strategy:
        #     print("MTL auto weighting...")
        #     self.weighting_strategy = getattr(weighting, self.weighting_strategy)(device=self.device, num_task=self.num_task)
        #     self.train_loss_buffer = None # should be defined or initiated in trainer/train.py
        #     self.train_loss_buffer_per_epoch = [] # should be reset after the last epoch finishes
        
        if self.is_train:                        
            self.init_training_settings()
            # if not dist.is_initialized():
            #     dist.init_process_group(
            #         backend='nccl',
            #         init_method='env://',
            #         world_size=opt['world_size'],
            #         rank=0,
            #         timeout=datetime.timedelta(hours=6)
            #     )
        # self.TTA = self.opt['val'].get('TTA', False) 
        # self.vi_out_dim = self.opt['network_g'].get('vi_out_dim', 3)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.loss_cls_dict = OrderedDict()
        # loss class name must start with the prefix "l_", which is defined in the logging function
        for loss_name, loss_ in train_opt.get('losses', {}).items():
            self.loss_cls_dict[f'l_{loss_name}'] = build_loss(loss_).to(self.device)
        # 如果loss_cls_dict为空，则抛出异常
        if len(self.loss_cls_dict) == 0:
            raise ValueError('No loss function provided.')
                    
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.ir = data['ir'].to(self.device)
        self.vi = data['vi'].to(self.device)
        self.ir_warped = data['ir_warped'].to(self.device)
        if 'vi_warped' in data:
            self.vi_warped = data['vi_warped'].to(self.device)
        self.disp = data['disp'].to(self.device)
        self.img_name_list = data['img_name']
        # 注意vi_warped可能是None（val或test阶段），使用前需判断一下
    def optimize_parameters(self, current_iter, current_epoch):
        self.optimizer_g.zero_grad() 
        # assert self.vi_warped != None
        l_total = 0
        loss_dict = OrderedDict()
        
        # print(self.ir.shape)
        b,c,h,w = self.ir.shape
        ir_stack = torch.cat([self.ir_warped, self.ir])
        vi_stack = torch.cat([self.vi, self.vi_warped])
        with autocast(enabled=self.use_amp):
            ir_pred, ir_disp_pred, vi_pred, vi_disp_pred = self.net_g(ir_stack, vi_stack, flow_direction='bi')            
        
            ir_reg, ir_warped_fake = torch.split(ir_pred, b, dim=0)
            vi_warped_fake, vi_reg = torch.split(vi_pred, b, dim=0)
            # forward表示用dataset类中的disp去做gird_sample，即预测场要接近GT场，而不是要估计GT disp的inverse/reverse变换
            ir2vi_forward_disp = ir_disp_pred[b:]
            vi2ir_forward_disp = vi_disp_pred[:b]
            # print(ir2vi_forward_disp.shape)
            
            # if isinstance(self.net_g, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            #     goodmask, goodmask_inv = self.net_g.module.generate_mask(self.disp)
            # 如果是dynamic_shape，这里应该动态生成grid，这里不处理了
            flow = self.grid + self.disp
            goodmask = torch.logical_and(flow >= -1, flow <= 1)
            goodmask = torch.logical_and(goodmask[..., 0], goodmask[..., 1]).unsqueeze(1) * 1.0
            AP = torch.nn.AvgPool2d(5, stride=1, padding=2)
            for _ in range(2):
                goodmask = (AP(goodmask) > 0.3).float()
            
            flow = self.grid - self.disp
            goodmask_inv = F.grid_sample(goodmask, flow, align_corners=False)
            
            loss_dict['l_Photometric'] = self.loss_cls_dict['l_Photometric'](self.ir_warped, ir_warped_fake, goodmask) \
                + self.loss_cls_dict['l_Photometric'](ir_reg, self.ir, goodmask * goodmask_inv) \
                + self.loss_cls_dict['l_Photometric'](self.vi_warped, vi_warped_fake, goodmask) \
                + self.loss_cls_dict['l_Photometric'](vi_reg, self.vi, goodmask * goodmask_inv)
            loss_dict['l_EndPoint'] = self.loss_cls_dict['l_EndPoint'](self.ir_warped, vi_warped_fake, vi2ir_forward_disp, self.disp) \
                + self.loss_cls_dict['l_EndPoint'](self.vi_warped, ir_warped_fake, ir2vi_forward_disp, self.disp)
            loss_dict['l_DefReg'] = self.loss_cls_dict['l_DefReg'](ir2vi_forward_disp.permute(0,3,1,2)) + self.loss_cls_dict['l_DefReg'](vi2ir_forward_disp.permute(0,3,1,2))
            l_total = loss_dict['l_Photometric'] + loss_dict['l_EndPoint'] + loss_dict['l_DefReg']
            loss_dict['l_total'] = l_total
            
        
        if current_iter%self.opt.get('display_intermediate_freq', 1000) == 0:
            os.makedirs(os.path.join(self.opt['path']['experiments_root'], 'display'), exist_ok=True)
            # display intermidate result in training
            img_display = torch.cat([self.ir[0:1, 0:1], 
                                    self.ir_warped[0:1, 0:1],
                                    ir_reg[0:1, 0:1],
                                    (self.ir[0:1, :]-ir_reg[0:1, :]).abs().mean(dim=1, keepdim=True)])
            img_display = torchvision.utils.make_grid(
                img_display, nrow=img_display.size(0)//2)
            torchvision.utils.save_image(img_display, os.path.join(self.opt['path']['experiments_root'], 'display', f'{current_iter}.png'))
        
        
        
        # if self.weighting_strategy is not None:
        #     loss_multi_task = torch.stack([l_fusion, l_seg]).squeeze(0).to(self.device)
        #     # TLAW: total loss after weighting
        #     l_total = self.weighting_strategy.backward(loss_multi_task, current_epoch, self.train_loss_buffer)
        #     loss_dict['l_TLAW'] = l_total
        #     self.train_loss_buffer_per_epoch.append(np.array([l_fusion.item(), l_seg.item()]))
            
        if self.use_amp:
            self.amp_scaler.scale(l_total).backward()
            self.amp_scaler.unscale_(self.optimizer_g) # 在梯度裁剪前先unscale梯度
            if self.opt['train'].get('use_grad_clip', False):
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt['train']['max_grad_norm'])
            scale_factor = self.amp_scaler.get_scale()
            if scale_factor < 0.001:
                logger = get_root_logger()
                logger.info(f'Current scale factor is too small: {self.amp_scaler.get_scale():.6f}, meaning that the loss/gradient is too large')
            self.amp_scaler.step(self.optimizer_g)
            self.amp_scaler.update()        
        else:
            l_total.backward()
            if self.opt['train'].get('use_grad_clip', False):
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt['train']['max_grad_norm'], error_if_nonfinite=False)
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)                
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.fusion, self.seg_result = self.net_g_ema(self.ir, self.vi)            
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.fusion, self.seg_result = self.net_g(self.ir, self.vi)
            self.net_g.train()
    
    # adapted from SegNeXt-main\mmseg\models\segmentors\encoder_decoder.py EncoderDecoder.slide_inference()
    def test_slide(self, ir, vi):
        # If h_crop > h_img or w_crop > w_img, the small patch will be used to decode without padding.
        if self.opt['datasets'].get('train', None) is not None:
            if isinstance(self.opt['datasets']['train'].get('gt_size'), list):
                w_stride, h_stride = self.opt['datasets']['train']['gt_size']
            elif isinstance(self.opt['datasets']['train'].get('gt_size'), int):
                w_stride = h_stride = self.opt['datasets']['train']['gt_size']
        else:
            w_stride = h_stride = self.opt.get('infer_slide_max_size', 640)
        h_crop = h_stride   # crop可以大于stride，patch之间会有一定重叠，可能会带来更好的效果，但是会增加推理时间
        w_crop = w_stride
        h_stride -= self.opt.get('overlap_slide_inference', 0)
        w_stride -= self.opt.get('overlap_slide_inference', 0)
        bs, _, h_img, w_img = vi.shape    
        out_channels = self.opt['network_g'].get("vi_out_dim", 3)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds_fusion = torch.zeros(bs, out_channels, h_img, w_img)
        preds_seg = torch.zeros(bs, int(self.opt['seg_num_class']), h_img, w_img)
        count_mat = torch.zeros(bs, 1, h_img, w_img)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()            
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    vi_patch = vi[:, :, y1:y2, x1:x2]
                    ir_patch = ir[:, :, y1:y2, x1:x2]
                    with torch.no_grad():
                        pred_fusion, pred_seg = self.net_g_ema(ir_patch, vi_patch)
                    preds_fusion[:, :, y1:y2, x1:x2] += pred_fusion.to('cpu') # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
                    preds_seg[:, :, y1:y2, x1:x2] += pred_seg.to('cpu')
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0, "might encounter zero dividing error"
            preds_fusion = preds_fusion / count_mat
            self.fusion = preds_fusion.to(self.device)
            preds_seg /= count_mat
            self.seg_result = preds_seg.to(self.device)
            # self.net_g_ema.train()            
        else:
            self.net_g.eval()            
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    vi_patch = vi[:, :, y1:y2, x1:x2]
                    ir_patch = ir[:, :, y1:y2, x1:x2]
                    with torch.no_grad():
                        pred_fusion, pred_seg = self.net_g(ir_patch, vi_patch)
                    preds_fusion[:, :, y1:y2, x1:x2] += pred_fusion.to('cpu') # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
                    preds_seg[:, :, y1:y2, x1:x2] += pred_seg.to('cpu')
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0, "might encounter zero dividing error"
            preds_fusion = preds_fusion / count_mat
            self.fusion = preds_fusion.to(self.device)
            preds_seg /= count_mat
            self.seg_result = preds_seg.to(self.device)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    # multi-scale inference/self-ensemble test/test time augmentation
    # refer to mmseg's MultiScaleFlipAug
    # 注意也要处理infer_slide_max_size的逻辑
    # 进行以下操作的组合：
    # img_ratios = [0.75, 1.0, 1.25, 1.5, 1.75]
    # vflip
    # hflip
    # rot90
    def test_TTA(self):
        flip_aug = [False, True] if self.opt['val']['TTA_pipeline']['flip_aug'] else [False]
        hw_equal = self.ir.shape[2] == self.ir.shape[3]
        rot90_aug = [False, True] if self.opt['val']['TTA_pipeline']['rot90_aug'] and hw_equal else [False]
        flip_direction = self.opt['val']['TTA_pipeline']['flip_direction']        
        img_ratios = self.opt['val']['TTA_pipeline']['img_ratios']

        results_fusion = []
        results_seg = []        
        
        for ratio in img_ratios:
            vi_aug = torch.nn.functional.interpolate(self.vi, scale_factor=ratio, mode='bilinear', align_corners=False)
            ir_aug = torch.nn.functional.interpolate(self.ir, scale_factor=ratio, mode='bilinear', align_corners=False)
            max_img_size = max(ir_aug.shape[2:])

            for flip in flip_aug:
                for direction in flip_direction:
                    for rot in rot90_aug:
                        # vi_aug = vi_aug.clone()
                        # ir_aug = ir_aug.clone()

                        if flip:
                            if direction == 'horizontal':
                                vi_aug = torch.flip(vi_aug, dims=[3])
                                ir_aug = torch.flip(ir_aug, dims=[3])
                            elif direction == 'vertical':
                                vi_aug = torch.flip(vi_aug, dims=[2])
                                ir_aug = torch.flip(ir_aug, dims=[2])

                        if rot:
                            vi_aug = torch.rot90(vi_aug, k=1, dims=[2, 3])
                            ir_aug = torch.rot90(ir_aug, k=1, dims=[2, 3])
                        
                        if max_img_size <= self.opt.get('infer_slide_max_size', 640):
                            if hasattr(self, 'net_g_ema'):
                                self.net_g_ema.eval()
                                with torch.no_grad():
                                    fusion_pred, seg_pred = self.net_g(ir_aug, vi_aug)
                            else:
                                self.net_g.eval()
                                with torch.no_grad():
                                    fusion_pred, seg_pred = self.net_g(ir_aug, vi_aug)
                                self.net_g.train()
                        else:
                            self.test_slide(ir_aug, vi_aug)
                            fusion_pred = self.fusion
                            seg_pred = self.seg_result
                        if rot:
                            fusion_pred = torch.rot90(fusion_pred, k=-1, dims=[2, 3])
                            seg_pred = torch.rot90(seg_pred, k=-1, dims=[2, 3])

                        if flip:
                            if direction == 'horizontal':
                                fusion_pred = torch.flip(fusion_pred, dims=[3])
                                seg_pred = torch.flip(seg_pred, dims=[3])
                            elif direction == 'vertical':
                                fusion_pred = torch.flip(fusion_pred, dims=[2])
                                seg_pred = torch.flip(seg_pred, dims=[2])

                        fusion_pred = torch.nn.functional.interpolate(fusion_pred, size=self.vi.shape[2:], mode='bilinear', align_corners=False)
                        seg_pred = torch.nn.functional.interpolate(seg_pred, size=self.vi.shape[2:], mode='bilinear', align_corners=False)

                        results_fusion.append(fusion_pred)
                        results_seg.append(seg_pred)

        self.fusion = torch.mean(torch.stack(results_fusion), dim=0)
        self.seg_result = torch.mean(torch.stack(results_seg), dim=0)
        
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_best_metric = ['psnr']):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_best_metric)
        # dist.barrier()

    # 逐照片验证，batch_size=1，这是basicsr框架的一个缺陷，build_dataloader()里对于val、test阶段的batch_size写死为1
    # 导致验证时速度较慢
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_best_metric = ['psnr']):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            else:
                # zero self.metric_results        
                self.metric_results = {metric: 0 for metric in self.metric_results}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)            

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
            
        torch.cuda.empty_cache()        
        self.net_g.eval()
        for idx, val_data in enumerate(dataloader):            
            self.feed_data(val_data)
            
            # max_img_size = max(self.ir.shape[2:])
            # if self.TTA:
            #     self.test_TTA()
            # else:
            #     if max_img_size <= self.opt.get('infer_slide_max_size', 640):
            #         self.test()
            #     else:
            #         self.test_slide(self.ir, self.vi)
            
            # print(dataloader.dataset.phase)
            
            with torch.no_grad():
                # ir_pred, ir_disp_pred, _, vi_disp_pred = self.net_g(self.ir_warped, self.vi, flow_direction='bi')
                ir_pred, ir_disp_pred = self.net_g(self.ir_warped, self.vi)
            # print(torch.isnan(self.vi).any(), torch.isnan(self.ir_warped).any())
            # print(self.vi.shape, self.ir.shape)
    
            # print(ir_pred.shape, ir_disp_pred.shape, vi_pred.shape) 
            metric_data['src'] = ir_pred
            metric_data['tgt'] = self.ir
            metric_data['gt_flow'] = self.disp
            metric_data['pred_flow'] = ir_disp_pred
            # metric_data['pred_inv_flow'] = vi_disp_pred
            
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items(): 
                    self.metric_results[name] += calculate_metric(metric_data, opt_)                        
            if save_img:
                # 测试或验证阶段若有多个数据集，应该在save_img时将文件名加上dataset_name！
                save_path = osp.join(self.opt['path']['visualization'], dataset_name, self.img_name_list[0])
                imwrite(tensor2img([ir_pred], rgb2bgr=True), save_path)
                                
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()       

        self.net_g.train()
        
        
        best_updated = []
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                updated = self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter, save_best_metric = save_best_metric)
                if updated is not None:
                    best_updated.append(updated)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            if len(best_updated) and self.is_train:
                # 如果在训练阶段有多个ValSet，应该在save_best()里的文件名加上dataset_name！
                # 这里暂时不改，留个坑                
                self.save_best(current_iter, save_best_metric=best_updated)
    
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name} @ {current_iter} iter\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

