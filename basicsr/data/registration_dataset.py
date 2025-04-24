from torch.utils import data as data
import torchvision
from torchvision.transforms.functional import normalize, to_tensor, resize
import torch
import torch.nn.functional as F
# from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
# from basicsr.data.transforms import augment, paired_random_crop
# from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY

import os
# from natsort import natsorted
from PIL import Image
import random


import kornia
import kornia.geometry.transform as KGT
import kornia.utils as KU
import kornia.filters as KF

def randflow(img,angle=8,trans=0.08,ratio=1,sigma=15,base=500, blur_iter=6):
    h,w=img.shape[2],img.shape[3]
    # affine
    if not base is None:
        base_scale = base/torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # angle = max(w,h)/base*angle
        #print('hello')
    else:
        base_scale = 1
    rand_angle = (torch.rand(1)*2-1)*angle
    rand_trans = (torch.rand(1,2)*2-1)*trans
    M = KGT.get_affine_matrix2d(translations=rand_trans,center=torch.zeros(1,2),scale=torch.ones(1,2),angle=rand_angle)
    M = M.inverse()
    grid = KU.create_meshgrid(h,w).to(img.device)
    warp_grid = kornia.geometry.linalg.transform_points(M,grid)
    # warp_grid = grid
    #elastic
    disp = torch.rand([1,2,h,w])*2-1
    #disp = KF.gaussian_blur2d(disp,kernel_size=[(3*sigma)//2*2+1,(3*sigma)//2*2+1],sigma=[sigma,sigma])
    # for i in range(5):
    for _ in range(blur_iter-1):
        disp = KF.gaussian_blur2d(disp,kernel_size=((3*sigma)//2*2+1,(3*sigma)//2*2+1),sigma=(sigma,sigma))
    disp = KF.gaussian_blur2d(disp,kernel_size=((3*sigma)//2*2+1,(3*sigma)//2*2+1),sigma=(sigma,sigma)).permute(0,2,3,1)*ratio

    disp = (disp+warp_grid-grid)*base_scale
    trans_grid = grid+disp
    mask = trans_grid<-1
    mask = torch.logical_or(trans_grid>1,mask)
    # t = KF.gaussian_blur2d(disp.permute(0,3,1,2),kernel_size=[9,9],sigma=[3,3])
    # img_p = F.unfold(t,kernel_size=11,dilation=1,padding=5,stride=1).reshape(1,2,-1,h,w)
    # local_var = img_p[...,10:-10,10:-10].var(dim=2).clamp(1e-6).mean()
    # # global_var = img[...,10:-10,10:-10].var(dim=[-1,-2]).clamp(1e-7).mean()
    # print(local_var)
    return trans_grid, disp, mask

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
    return im_ts.unsqueeze(0)

# D:\学习\电子课题组\论文\配准与融合\配准\UMF-CMGR\UMF-CMGR-main\functions\elastic_transform.py
# 傻逼了没考虑到不同数据集图片尺寸不一样，干脆直接暴力把RoadScene都resize到480*640吧
@DATASET_REGISTRY.register()
class RegDataset(data.Dataset):
    def __init__(self, opt):
        super(RegDataset, self).__init__()
        self.opt = opt        
        self.phase = self.opt['phase']
        
        # print(opt)
        if self.phase == 'train':
            if isinstance(self.opt['gt_size'], list):
                assert len(self.opt['gt_size']) == 2, "gt_size must be a list of two integers: [width, height]"
                self.patch_w, self.patch_h = self.opt['gt_size']
            elif isinstance(self.opt['gt_size'], int):
                self.patch_w = self.patch_h = self.opt['gt_size']
            self.random_crop = torchvision.transforms.RandomCrop((self.patch_h, self.patch_w))
        
        self.ir_folder = []
        self.vi_folder = []
        self.dataset_len_list = []
        if self.phase != 'train':
            self.ir_warp_folder = []
            self.flow_folder = []
            self.flow_format = opt.get('flow_format', '.mat')
        self.ir_list = []
        for dataset in opt['dataset_list']:
            ir_dir_path = os.path.join(dataset, opt['ir_dir_name'])
            self.ir_folder.append(ir_dir_path)
            self.ir_list.append(os.listdir(ir_dir_path))
            self.dataset_len_list.append(len(os.listdir(ir_dir_path)))
            self.vi_folder.append(os.path.join(dataset, opt['vi_dir_name']))
            if self.phase != 'train':
                self.ir_warp_folder.append(os.path.join(dataset, opt['ir_warp_dir_name']))
                self.flow_folder.append(os.path.join(dataset, opt['flow_dir_name']))
                # 这里加载的flow就是指位移矢量场displacement，而SuperFusion里保存的.mat格式flow好像指的是warped_grid（matlab加载后看它长什么样看出来的）
        
        self.dataset_len = sum(self.dataset_len_list)
                
        
    def __getitem__(self, index):
        dataset_idx = 0
        index_ = index
        for i in self.dataset_len_list:
            if index_ < i:
                break
            index_ -= i
            dataset_idx += 1
        img_name = self.ir_list[dataset_idx][index_]
        ir_path = os.path.join(self.ir_folder[dataset_idx], img_name)
        vi_path = os.path.join(self.vi_folder[dataset_idx], img_name)
        vi = imread(vi_path, label=False, vis_flag=True)
        ir = imread(ir_path, label=False, vis_flag=True)
        if self.phase != 'train':
            ir_warp_path = os.path.join(self.ir_warp_folder[dataset_idx], img_name)
            flow_path = os.path.join(self.flow_folder[dataset_idx], img_name+self.flow_format)
            ir_warped = imread(ir_warp_path, label=False, vis_flag=True)
            # torch.Size([1, 480, 640, 2])
            disp = torch.load(flow_path)
            return {'ir':ir.squeeze(0), 'vi':vi.squeeze(0), 'ir_warped':ir_warped.squeeze(0), 'disp':disp.squeeze(0), 'img_name':img_name}
        else:            
            vi_ir = torch.cat([vi, ir], dim=1)
            
            # 确保能crop到大小为gt_size的patch
            if vi_ir.shape[-2] < self.patch_h or vi_ir.shape[-1] < self.patch_w:
                if vi_ir.shape[-2]/self.patch_h > vi_ir.shape[-1]/self.patch_w:
                    vi_ir = resize(vi_ir, (int(vi_ir.shape[-2]/vi_ir.shape[-1]*self.patch_w), self.patch_w), antialias=True)
                else:
                    vi_ir = resize(vi_ir, (self.patch_h, int(vi_ir.shape[-1]/vi_ir.shape[-2]*self.patch_h)), antialias=True)
            if self.opt.get('use_rot', False) and self.patch_h == self.patch_w:
                    vi_ir = torch.rot90(vi_ir, random.randint(0, 3), dims=(-2, -1))
            if self.opt.get('use_transpose', False) and self.patch_h == self.patch_w and random.random() < 0.5:
                vi_ir = vi_ir.transpose(-2, -1)
            if self.opt.get('use_flip', False) and random.random() < 0.5:
                vi_ir = vi_ir.flip(-2) # flip vertically
                if random.random() < 0.5:
                    vi_ir = vi_ir.flip(-1) # flip horizontally
            warped_grid, disp, _ = randflow(vi_ir)
            # disp.shape: [1,h,w,2]
            vi_ir_warped = F.grid_sample(vi_ir, warped_grid, align_corners=False, mode='bilinear')
            patch = torch.cat([vi_ir, vi_ir_warped, disp.permute(0,3,1,2)], dim=1)
            patch = self.random_crop(patch)
            vi, ir, vi_warped, ir_warped, disp = torch.split(patch, [3,3,3,3,2], dim=1)
            h,w = vi.shape[2], vi.shape[3]
            scale = (torch.FloatTensor([w,h]).unsqueeze(0).squeeze(0)-1)/(torch.FloatTensor([self.random_crop.size[1], self.random_crop.size[0]]) - 1)
            # 因为disp对于完整图像left to right, top to bottom的值分布在[-1,1]范围内
            # 所以crop后乘以scale是关于cropped patch坐标系上的归一化disp
            disp = disp.permute(0,2,3,1)*scale
            # return ir.squeeze(0), vi.squeeze(0), ir_warped.squeeze(0), vi_warped.squeeze(0), disp.squeeze(0), img_name
            return {'ir':ir.squeeze(0), 'vi':vi.squeeze(0), 'ir_warped':ir_warped.squeeze(0), 'vi_warped':vi_warped.squeeze(0),'disp':disp.squeeze(0), 'img_name':img_name}

    def __len__(self):
        return self.dataset_len