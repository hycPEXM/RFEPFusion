from torch.utils import data as data
import torchvision
from torchvision.transforms.functional import normalize, to_tensor, resize
import torch
# from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
# from basicsr.data.transforms import augment, paired_random_crop
# from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY

import os
from natsort import natsorted
from PIL import Image
import random

def imread(path, label=False, vis_flag=True):
    if label:
        img = Image.open(path)
        # im_ts = to_tensor(img).unsqueeze(0) * 255
        im_ts = to_tensor(img)*255
    else:
        if vis_flag:  # visible images; RGB channel
            img = Image.open(path).convert('RGB')
            # im_ts = to_tensor(img).unsqueeze(0)
            im_ts = to_tensor(img)
        else:  # infrared images single channel 
            img = Image.open(path).convert('L') 
            # im_ts = to_tensor(img).unsqueeze(0)
            im_ts = to_tensor(img)
    return im_ts

# train and validation
@DATASET_REGISTRY.register()
class MSRSDataset(data.Dataset):
    def __init__(self, opt):
        super(MSRSDataset, self).__init__()
        self.opt = opt
        
        # mask is only used in training
        self.mask_folder = None
        self.enhaced_folder = None
        self.random_crop = None
        self.patch_w = None
        self.patch_h = None
        if self.opt['phase'] == 'train':            
            self.mask_folder = os.path.join(opt['dataroot'], opt['mask_dir_name'])     
            if isinstance(self.opt['gt_size'], list):
                assert len(self.opt['gt_size']) == 2, "gt_size must be a list of two integers: [width, height]"
                self.patch_w, self.patch_h = self.opt['gt_size']
            elif isinstance(self.opt['gt_size'], int):
                self.patch_w = self.patch_h = self.opt['gt_size']
            self.random_crop = torchvision.transforms.RandomCrop((self.patch_h, self.patch_w))
            self.enhaced_folder = os.path.join(opt['dataroot'], opt['vi_enhanced_dir_name'])
        self.vi_folder = os.path.join(opt['dataroot'], opt['vi_dir_name'])
        self.ir_folder = os.path.join(opt['dataroot'], opt['ir_dir_name'])
        self.seg_folder = os.path.join(opt['dataroot'], opt['seg_dir_name'])

        self.ir_list = natsorted(os.listdir(self.ir_folder))        
        
    def __getitem__(self, index):
        
        img_name = self.ir_list[index]
        vi_path = os.path.join(self.vi_folder, img_name)
        ir_path = os.path.join(self.ir_folder, img_name)
        seg_path = os.path.join(self.seg_folder, img_name)
        vi = imread(vi_path, label=False, vis_flag=True)
        ir = imread(ir_path, label=False, vis_flag=False)
        seg = imread(seg_path, label=True, vis_flag=False)        
        if self.opt['phase'] == 'train':
            mask_path = os.path.join(self.mask_folder, img_name)
            mask = imread(mask_path, label=False, vis_flag=False)
            enhanced_path = os.path.join(self.enhaced_folder, img_name)
            enhanced = imread(enhanced_path, label=False, vis_flag=True)
                        
            to_augment = torch.cat([vi, enhanced, ir, seg, mask], dim=0)
            
            # 缩放统一采用等比例缩放：keep_ratio=True
            if isinstance(self.opt.get('resize_ratio_range'), list):
                resize_ratio = random.uniform(self.opt['resize_ratio_range'][0], self.opt['resize_ratio_range'][1])
                to_augment = resize(to_augment, (int(to_augment.shape[-2]*resize_ratio), int(to_augment.shape[-1]*resize_ratio)), antialias=True)
            # 确保能crop到大小为gt_size的patch
            if to_augment.shape[-2] < self.patch_h or to_augment.shape[-1] < self.patch_w:
                if to_augment.shape[-2]/self.patch_h > to_augment.shape[-1]/self.patch_w:
                    to_augment = resize(to_augment, (int(to_augment.shape[-2]/to_augment.shape[-1]*self.patch_w), self.patch_w), antialias=True)
                else:
                    to_augment = resize(to_augment, (self.patch_h, int(to_augment.shape[-1]/to_augment.shape[-2]*self.patch_h)), antialias=True)
            to_augment = self.random_crop(to_augment)
            if self.opt.get('use_rot', False) and self.patch_h == self.patch_w:
                to_augment = torch.rot90(to_augment, random.randint(0, 3), dims=(-2, -1))
            if self.opt.get('use_transpose', False) and self.patch_h == self.patch_w and random.random() < 0.5:
                to_augment = to_augment.transpose(-2, -1)
            if self.opt.get('use_flip', False) and random.random() < 0.5:
                to_augment = to_augment.flip(-2) # flip vertically
                if random.random() < 0.5:
                    to_augment = to_augment.flip(-1) # flip horizontally
            
            vi, enhanced, ir, seg, mask = torch.split(to_augment, [3, 3, 1, 1, 1], dim=0)
            
            seg = seg.long()
            return {
                'ir': ir,
                'vi': vi,
                'seg': seg,
                'mask': mask,
                'enhanced': enhanced,
                'img_name': img_name
            }
        else:
            seg = seg.long()
            return {
                'ir': ir,
                'vi': vi,
                'seg': seg,
                'img_name': img_name
            }                    

    def __len__(self):
        return len(self.ir_list)

# For testing
@DATASET_REGISTRY.register()
class FusionSegDataset(data.Dataset):
    def __init__(self, opt):
        super(MSRSDataset, self).__init__()
        self.opt = opt
                        
        self.vi_folder = os.path.join(opt['dataroot'], opt['vi_dir_name'])
        self.ir_folder = os.path.join(opt['dataroot'], opt['ir_dir_name'])

        self.ir_list = natsorted(os.listdir(self.ir_folder))        
    
    # 不需要padding
    def __getitem__(self, index):
        
        img_name = self.ir_list[index]
        vi_path = os.path.join(self.vi_folder, img_name)
        ir_path = os.path.join(self.ir_folder, img_name)
        
        vi = imread(vi_path, label=False, vis_flag=True)
        ir = imread(ir_path, label=False, vis_flag=False)
                           
        return {
            'ir': ir,
            'vi': vi,
            'img_name': img_name
        }                    

    def __len__(self):
        return len(self.ir_list)