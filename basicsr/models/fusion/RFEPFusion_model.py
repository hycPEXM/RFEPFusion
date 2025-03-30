import torch
from collections import OrderedDict
from os import path as osp
# import os
import torch.distributed
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, scandir
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel

from basicsr.data.fusion_utils import RGB2YCrCb, YCbCr2RGB

# from .auto_weighting import *
import basicsr.models.fusion.auto_weighting as weighting

try :
    from torch.cuda.amp import autocast, GradScaler
    load_amp = True
except:
    load_amp = False

import random
import cv2
import numpy as np

# MSRS提供的调色板
def get_palette():
    unlabelled = [0, 0, 0]
    car = [64, 0, 128]
    person = [64, 64, 0]
    bike = [0, 128, 192]
    curve = [0, 0, 192]
    car_stop = [128, 128, 0]
    guardrail = [64, 64, 128]
    color_cone = [192, 128, 128]
    bump = [192, 64, 0]
    palette = np.array(
        [
            unlabelled,
            car,
            person,
            bike,
            curve,
            car_stop,
            guardrail,
            color_cone,
            bump,
        ]
    )
    return palette

# def visualize(save_name, label):
#     palette = get_palette()
#     pred = label
#     img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
#     for cid in range(1, int(label.max()+1)):
#         img[pred == cid] = palette[cid]
#     img = Image.fromarray(np.uint8(img))
#     img.save(save_name)

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
        
        # def check_nan_hook(module, input, output):
        #     if isinstance(output, tuple):
        #         outputs = output
        #     else:
        #         outputs = [output]
        #     for idx, tensor in enumerate(outputs):
        #         if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
        #             # 打印完整路径和具体输出位置（如元组中的第几个元素）
        #             print(f"NaN detected in module: {module.__class__.__name__} (path: {module._get_name()})")
        #             print(f"max: {torch.max(tensor)}, min: {torch.min(tensor)}")
        #             print(f"Output position: {idx} of output tuple")
        #             print(f"Shape of NaN tensor: {tensor.shape}")
        #             print("-------------------------------")
        #             # 可选：抛出异常终止训练（调试时建议）
        #             # raise RuntimeError("NaN detected, stopping training.")
        # for name, layer in self.net_g.named_modules():
        #     # if isinstance(layer, nn.Linear):  # 只对线性层注册钩子
        #     #     layer.register_forward_hook(check_nan_hook)
        #     layer.register_forward_hook(check_nan_hook)
            
        # for name, param in self.net_g.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"NaN detected in parameter: {name}\n\n")
        
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if opt['dist'] or opt['num_gpu']>1:
            print("Turning BatchNorm to SyncBN!")
            self.net_g = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_g)
        
        self.num_task = 2 # fusion + segmentation
        
        self.weighting_strategy = opt['train'].get('MTL_auto_weighting')
                        
        # ‌Automatic Mixed Precision Training
        self.use_amp = opt.get('use_amp', False) and load_amp        
        if self.use_amp:
            self.amp_scaler = GradScaler(enabled=self.use_amp)
            print('Using Automatic Mixed Precision')
        else:
            print('Not using Automatic Mixed Precision')                
        
        # 目前支持amp与weighting一起使用，但不推荐
        # if self.weighting_strategy and self.use_amp:
        #     raise ValueError('auto weighting currently can\'t used with amp training') 
        
        if self.weighting_strategy:
            self.weighting_strategy = getattr(weighting, self.weighting_strategy)(device=self.device, num_task=self.num_task)
            self.train_loss_buffer = None # should be defined or initiated in trainer/train.py
            self.train_loss_buffer_per_epoch = [] # should be reset after the last epoch finishes
        
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
        self.TTA = self.opt['val'].get('TTA', False)        

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
        self.vi = data['vi'].to(self.device)
        self.ir = data['ir'].to(self.device)        
        if 'seg' in data:
            self.seg_label = data['seg'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        if 'enhanced' in data:
            self.enhanced = data['enhanced'].to(self.device)
        if 'img_name' in data:
            self.img_list = data['img_name']

    def optimize_parameters(self, current_iter, current_epoch):
        """
        RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by 
        making sure all `forward` function outputs participate in calculating loss. 
        If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
        Parameter indices which did not receive grad for rank 0: 694 695
        In addition, you can set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information about which particular parameters did not receive gradient on this rank as part of this error
        """
        # print(list(self.net_g.parameters())[694])
        # print(list(self.net_g.state_dict().keys())[694])
        # print(list(self.net_g.state_dict().keys())[695])
        # 通过下面的代码定位到是下面的模块出了问题：
        # module.conv_dealign_stage2_4.2.weight
        # module.conv_dealign_stage2_4.2.bias
        # optim_params_keyname = []
        # for k, v in self.net_g.named_parameters():
        #     if v.requires_grad:
        #         optim_params_keyname.append(k)
        # print(optim_params_keyname[694])
        # print(optim_params_keyname[695])
        
        # max_value = float('-inf')
        # min_value = float('inf')

        # # 遍历模型的所有参数
        # for param in self.net_g.parameters():
        #     # 更新最大值
        #     current_max = torch.max(param).item()
        #     if current_max > max_value:
        #         max_value = current_max

        #     # 更新最小值
        #     current_min = torch.min(param).item()
        #     if current_min < min_value:
        #         min_value = current_min
        # # 打印结果
        # print(f"模型参数的最大值: {max_value}")
        # print(f"模型参数的最小值: {min_value}")
        
        self.optimizer_g.zero_grad()        
        l_total = 0
        l_fusion = 0
        l_seg = 0
        loss_dict = OrderedDict()
        with autocast(enabled=self.use_amp):
            # 混合精度训练的with语句块下需要进行模型前向计算和损失计算
            self.fusion, self.seg_result = self.net_g(self.ir, self.vi)
            
            # if torch.isnan(self.ir).any().item() or torch.isnan(self.vi).any().item():
            #     print("NaN detected in model input!")
            #     # if self.opt['rank'] == 0:
            #     #     import pdb
            #     #     pdb.set_trace()
            # if (self.ir>1+1e-6).any().item() or (self.ir<-1e-6).any().item() or (self.vi>1+1e-6).any().item() or (self.vi<-1e-6).any().item():
            #     print("Value out of range in model input!")
            # if torch.isnan(self.fusion).any().item() or torch.isnan(self.seg_result).any().item():
            #     print("NaN detected in model output!", self.opt['rank'])                                              
            #     # if self.opt['rank'] == 0:
            #     #     print("fusion:", torch.isnan(self.fusion).any())
            #     #     print("seg_result:", torch.isnan(self.seg_result).any())  
            #         # max_value = float('-inf')
            #         # min_value = float('inf')
            #         # for name, param in self.net_g.named_parameters():
            #         #     if param.max().item() > max_value:
            #         #         max_value = param.max().item()
            #         #     if param.min().item() < min_value:
            #         #         min_value = param.min().item()
            #         #     if torch.isnan(param).any():
            #         #         print(f"NaN detected in parameter: {name}\n\n")
            #         # print(f"模型参数的最大值: {max_value}")
            #         # print(f"模型参数的最小值: {min_value}")
            #         # max_value = float('-inf')
            #         # min_value = float('inf')
            #         # for name, param in self.net_g.module.vi_encoder.named_parameters():
            #         #     if param.max().item() > max_value:
            #         #         max_value = param.max().item()
            #         #     if param.min().item() < min_value:
            #         #         min_value = param.min().item()                    
            #         # print(f"vi_encoder模型参数的最大值: {max_value}")
            #         # print(f"vi_encoder模型参数的最小值: {min_value}")
            #         # max_value = float('-inf')
            #         # min_value = float('inf')
            #         # for name, param in self.net_g.module.ir_encoder.named_parameters():
            #         #     if param.max().item() > max_value:
            #         #         max_value = param.max().item()
            #         #     if param.min().item() < min_value:
            #         #         min_value = param.min().item()                    
            #         # print(f"ir_encoder模型参数的最大值: {max_value}")
            #         # print(f"ir_encoder模型参数的最小值: {min_value}")
            #         # self.net_g.module.forward_debug(self.ir, self.vi)
            #         # import time; time.sleep(500)
            # #         import pdb
            # #         pdb.set_trace()
            
            # NAN_flag = False
            # if torch.isnan(self.fusion).any().item() or torch.isnan(self.seg_result).any().item():
            #     for name, param in self.net_g.module.named_parameters():
            #         if torch.isnan(param).any():
            #             print(f"NaN detected in parameter: {name}")
            #     print("seg_result:", torch.isnan(self.seg_result).any(), self.img_list)
            #     print("fusion:", torch.isnan(self.fusion).any(), self.img_list)
            #     # self.net_g.module.forward_debug(self.ir, self.vi)
            #     # import time; time.sleep(10000)
            #     NAN_flag = True
            
            # # Share NAN_flag across different GPU processes
            # NAN_flag_tensor = torch.tensor(NAN_flag, dtype=torch.bool, device=self.device)
            # torch.distributed.all_reduce(NAN_flag_tensor, op=torch.distributed.ReduceOp.SUM)
            # NAN_flag = NAN_flag_tensor.item() > 0

            # if NAN_flag:
            #     # self.fusion, self.seg_result = self.net_g(self.ir, self.vi)
            #     # 直接跳过NAN的iteration
            #     with torch.no_grad():
            #         self.fusion, self.seg_result = self.net_g(self.ir, self.vi)
            #     return
            
            # # torch.distributed.barrier()            
            
            # loss计算时的参数命名：ir, Y_vi, Y_fusion, vi, fusion, seg_result, seg_label
            Y_vi, _, _ = RGB2YCrCb(self.vi)
            Y_enhanced, _, _ = RGB2YCrCb(self.enhanced)
            Y_fusion, _, _ = RGB2YCrCb(self.fusion)
            loss_kwargs = {
                'ir': self.ir,
                'Y_vi': Y_vi,
                'Y_fusion': Y_fusion,
                'Y_enhanced': Y_enhanced,
                'enhanced': self.enhanced,
                'vi': self.vi,
                'fusion': self.fusion,
                'seg_result': self.seg_result,                
                'seg_label': self.seg_label,
                'mask_person': self.mask
            }
            for loss_name, loss_cls in self.loss_cls_dict.items():
                # print(loss_name)
                # print(loss_kwargs['fusion'])
                loss = loss_cls(**loss_kwargs)
                l_total += loss
                if loss_name == 'l_seg':
                    l_seg = loss
                else: 
                    l_fusion += loss
                loss_dict[loss_name] = loss
                # print(loss_name,"---", loss)
            loss_dict['l_fusion'] = l_fusion
            loss_dict['l_total'] = l_total.clone()            
        
        # for k, t in loss_kwargs.items():
        #     if torch.isnan(t).any().item():
        #         print(k)
        
        # print(self.seg_label.shape)
        # print(self.mask.shape)
        # print(self.vi.shape)
        # print(loss_dict)
        # import time
        # time.sleep(2)
                      
        # for name, param in self.net_g.named_parameters():
        #     if torch.isnan(param).any():
        #         print(f"NaN detected in parameter: {name}")
        # import time
        # time.sleep(10)        
        
        if self.weighting_strategy is not None:
            loss_multi_task = torch.stack([l_fusion, l_seg]).squeeze(0).to(self.device)
            # TLAW: total loss after weighting
            l_total = self.weighting_strategy.backward(loss_multi_task, current_epoch, self.train_loss_buffer)
            loss_dict['l_TLAW'] = l_total
            self.train_loss_buffer_per_epoch.append(np.array([l_fusion.item(), l_seg.item()]))
            
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
        
        # print("log_dict:", self.log_dict)
        # print("after:", loss_dict)
        # time.sleep(5)
        
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
        bs, out_channels, h_img, w_img = vi.shape    
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
        
        # import time
        # metric_time = {}
        
        visualize_palette = get_palette()
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
                    self.test_slide(self.ir, self.vi)
                        
            self.vi = self.vi.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)      
            self.vi = cv2.cvtColor(self.vi, cv2.COLOR_RGB2GRAY)*255
            self.vi = self.vi.astype(np.float32)
            # self.vi = tensor2img([self.vi], rgb2bgr=False, out_type=np.float32)*255
            # print(self.ir.shape)
            self.ir = self.ir.squeeze(0).detach().cpu().numpy().squeeze(0)*255
            # self.ir = cv2.cvtColor(self.ir, cv2.COLOR_RGB2GRAY)*255
            self.ir = self.ir.astype(np.float32) 
            # self.ir = tensor2img([self.ir], rgb2bgr=False, out_type=np.float32)*255
            fusion_clone = self.fusion.clone()
            self.fusion = self.fusion.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            self.fusion = cv2.cvtColor(self.fusion, cv2.COLOR_RGB2GRAY)*255
            self.fusion = self.fusion.astype(np.float32)
            # self.fusion = tensor2img([self.fusion], rgb2bgr=False, out_type=np.float32)*255
            
            self.seg_result = torch.argmax(self.seg_result, dim=1, keepdim=True)
            
            metric_data['F'] = self.fusion
            metric_data['A'] = self.vi
            metric_data['B'] = self.ir
            metric_data['seg_result'] = self.seg_result
            # if hasattr(self, 'seg_label'):
            if 'seg' in val_data: 
                # if self.seg_label.ndim == 4:
                #     self.seg_label = self.seg_label.squeeze(1)
                metric_data['label'] = self.seg_label
                        
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    # start_time = time.time() 
                    if name == 'mIoUD':
                        self.metric_results[name] = calculate_metric(metric_data, opt_)
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
                    # if name not in metric_time:
                    #     metric_time[name] = time.time() - start_time
                    # else:
                    #     metric_time[name] += time.time() - start_time
                        
            if save_img:
                # 测试或验证阶段若有多个数据集，应该在save_img时将文件名加上dataset_name！
                if self.TTA:                        
                    save_fusion_path = osp.join(self.opt['path']['visualization'], dataset_name, 'TTA', 'fusion', f'{img_name}.png')
                    save_seg_path = osp.join(self.opt['path']['visualization'], dataset_name, 'TTA', 'seg', f'{img_name}.png')
                    save_seg_visualize_path = osp.join(self.opt['path']['visualization'], dataset_name, 'TTA', 'seg_visualize', f'{img_name}.png')
                else:
                    save_fusion_path = osp.join(self.opt['path']['visualization'], dataset_name, 'fusion', f'{img_name}.png')
                    save_seg_path = osp.join(self.opt['path']['visualization'], dataset_name, 'seg', f'{img_name}.png')
                    save_seg_visualize_path = osp.join(self.opt['path']['visualization'], dataset_name, 'seg_visualize', f'{img_name}.png')                
                imwrite(tensor2img([fusion_clone], rgb2bgr=True), save_fusion_path)                
                imwrite(tensor2img([self.seg_result], rgb2bgr=False, min_max=(0, 255)), save_seg_path)
                # 保存分割的可视化结果
                _, _, h, w = self.seg_result.shape
                seg_visualize = np.zeros((h, w, 3), dtype=np.uint8)
                seg_result_numpy = self.seg_result[0, 0].cpu().numpy()
                for cid, color in enumerate(visualize_palette):
                    seg_visualize[seg_result_numpy == cid] = color
                imwrite(seg_visualize, save_seg_visualize_path)
                                
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()       
                 
        # import pprint
        # pp = pprint.PrettyPrinter(indent=4)  # 设置缩进为 4 个空格
        # print('metric_time:')
        # pp.pprint(metric_time)
        
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
                # 如果在训练阶段有多个ValSet，应该在save_best()里的文件名加上dataset_name！
                # 这里暂时不改，留个坑
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
