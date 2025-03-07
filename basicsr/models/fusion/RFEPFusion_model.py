import torch
from collections import OrderedDict
from os import path as osp
# import os
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, scandir
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel

try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False

import random
import cv2
import numpy as np

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


@MODEL_REGISTRY.register()
class RFEPFusionNoRegModel(BaseModel):
    def __init__(self, opt):
        super(RFEPFusionNoRegModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if opt['dist']:
            print("Turning BatchNorm to SyncBN!")
            self.net_g = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_g)

        # ‌Automatic Mixed Precision Training
        self.use_amp = opt.get('use_amp', False) and load_amp
        self.amp_scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')                
        
        # self.mixing_flag = False
        if self.is_train:            
            # mixing_augs = self.opt['train'].get('mixing_augs', None)
            # if mixing_augs is not None:
            #     self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            #     if self.mixing_flag:
            #         mixup_beta = self.opt['train']['mixing_augs'].get(
            #             'mixup_beta', 1.2)
            #         use_identity = self.opt['train']['mixing_augs'].get(
            #             'use_identity', False)
            #         self.mixing_augmentation = Mixing_Augment(
            #             mixup_beta, use_identity, self.device)
            self.init_training_settings()

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
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.vi = data['vi'].to(self.device)
        self.ir = data['ir'].to(self.device)        
        if 'seg' in data:
            self.seg_label = data['seg'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        if 'enhanced' in data:
            self.enhanced = data['enhanced'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        
        l_total = 0
        loss_dict = OrderedDict()
        with  autocast(enabled=self.use_amp):
            # 混合精度训练的with语句块下需要进行模型前向计算和损失计算
            self.output = self.net_g(self.lq)            
            for loss_name, loss_cls in self.loss_cls_dict.items():
                #!!! 注意，在训练红外可见光融合的LLIE时，color_loss的self.gt应该换成self.lq！！！
                # 后来我感觉没必要加上这个color loss了
                if loss_name == 'l_color' and self.opt.get('use_lq_for_l_color', False):
                    loss = loss_cls(self.output, self.lq) 
                else:
                    loss = loss_cls(self.output, self.gt)                  
                l_total += loss
                loss_dict[loss_name] = loss
        
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
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.opt['train']['max_grad_norm'])
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.fusion, self.seg_result = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.fusion, self.seg_result = self.net_g(self.lq)
            self.net_g.train()
    
    # adapted from SegNeXt-main\mmseg\models\segmentors\encoder_decoder.py
    def test_slide(self):
        # If h_crop > h_img or w_crop > w_img, the small patch will be used to decode without padding.
        if self.opt['datasets'].get('train', None) is not None:
            w_stride, h_stride = self.opt['datasets']['train']['gt_size']
        else:
            w_stride = h_stride = self.opt.get('infer_slide_max_size', 640)
        h_crop = h_stride  # crop可以大于stride，patch之间会有一定重叠，可能会带来更好的效果，但是会增加推理时间
        w_crop = w_stride
        bs, out_channels, h_img, w_img = self.lq.shape
        # out_channels = 3        
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = torch.zeros(bs, out_channels, h_img, w_img)
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
                    lq_patch = self.lq[:, :, y1:y2, x1:x2]
                    with torch.no_grad():
                        pred = self.net_g_ema(lq_patch)
                    preds[:, :, y1:y2, x1:x2] += pred.to('cpu') # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0, "might encounter zero dividing error"
            preds = preds / count_mat
            self.output = preds.to(self.device)
            self.net_g_ema.train()            
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
                    lq_patch = self.lq[:, :, y1:y2, x1:x2]
                    with torch.no_grad():
                        pred = self.net_g(lq_patch)
                    preds[:, :, y1:y2, x1:x2] += pred.to('cpu') # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0, "might encounter zero dividing error"
            preds = preds / count_mat
            self.output = preds.to(self.device)
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

    def test_TTA(self):
        pass

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_best_metric = 'psnr'):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_best_metric)

    # 逐照片验证，batch_size=1，这是basicsr框架的一个缺陷，build_dataloader()里对于val、test阶段的batch_size写死为1
    # 导致验证时速度较慢
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_best_metric = 'psnr'):
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
        
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['img_name'][0]))[0]
            self.feed_data(val_data)
            
            max_img_size = max(self.ir.shape[2:])
            if self.TTA:
                self.test_TTA()
            else:
                if max_img_size <= self.opt.get('infer_slide_max_size', 640):
                    self.test()
                else:
                    self.test_slide()
                        
            self.vi = self.vi.squeeze(0).detach().cpu().numpy()
            self.vi = cv2.cvtColor(self.vi, cv2.COLOR_RGB2GRAY)*255
            self.vi = self.vi.astype(np.float32)
            self.ir = self.ir.squeeze(0).detach().cpu().numpy()
            self.ir = cv2.cvtColor(self.ir, cv2.COLOR_RGB2GRAY)*255
            self.ir = self.ir.astype(np.float32)          
            fusion_clone = self.fusion.clone()
            self.fusion = self.fusion.squeeze(0).detach().cpu().numpy()
            self.fusion = cv2.cvtColor(self.fusion, cv2.COLOR_RGB2GRAY)*255
            self.fusion = self.fusion.astype(np.float32)
            
            self.seg_result = torch.argmax(seg_result, dim=1, keepdim=True)
            
            metric_data['F'] = self.fusion
            metric_data['A'] = self.vi
            metric_data['B'] = self.ir
            metric_data['seg_result'] = self.seg_result
            if hasattr(self, 'seg_label'):
                metric_data['label'] = self.seg_label
            
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'mIoUD':
                        self.metric_results[name] = calculate_metric(metric_data, opt_)
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            
            # 暂不支持训练阶段有多个ValSet，因为更新best metric result时只考虑了一个ValSet，
            # 如果有多个ValSet，需要对每个ValSet都分别维护各自的best metric result
            if save_img:
                if self.TTA:                        
                    save_fusion_path = osp.join(self.opt['path']['visualization'], dataset_name, 'TTA', 'fusion', f'{img_name}.png')
                    save_seg_path = osp.join(self.opt['path']['visualization'], dataset_name, 'TTA', 'seg', f'{img_name}.png')
                else:
                    save_fusion_path = osp.join(self.opt['path']['visualization'], dataset_name, 'fusion', f'{img_name}.png')
                    save_seg_path = osp.join(self.opt['path']['visualization'], dataset_name, 'seg', f'{img_name}.png')
                imwrite(tensor2img([fusion_clone], rgb2bgr=True), save_fusion_path)
                imwrite(tensor2img([self.seg_result], rgb2bgr=False), save_seg_path)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()                

        best_updated_flag = False
        if with_metrics:
            for metric in self.metric_results.keys():
                if metric == 'mIoUD':
                    self.metric_results[metric] = self.metric_results[metric].meanIntersectionOverUnion()
                else:
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                best_updated_flag |= self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter, save_best_metric = save_best_metric)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            if best_updated_flag and self.is_train:
                self.save_best(current_iter, save_best_metric=save_best_metric)

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


# @MODEL_REGISTRY.register()
# class RFEPFusionModel(BaseModel):
#     def __init__(self, opt):
#         super(RFEPFusionModel, self).__init__(opt)
