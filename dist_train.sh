
##############################
# stage1: train LLIE
##############################
# stage 1.1: train_LLIE_vi_NTIRE2024_pretrained
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_vi_NTIRE2024_pretrained.yml --launcher pytorch 

# stage 1.2: train_LLIE_vi_LOL_pretrained
# CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_vi_LOL_pretrained.yml --launcher pytorch 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_vi_LOL_pretrained.yml --launcher pytorch
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_vi_LOL_pretrained_hyc_small.yml --launcher pytorch

# stage 1.3 train_LLIE_ir
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_ir.yml --launcher pytorch

# stage 1.4 train_LLIE_vi
# 训练老是中断，只能这样暂时应付一下了（同一个命令复制几次）
# /home/hongyuchen/master_thesis/RFEPFusion/experiments/RFEPFusion_LLIE_vi/best_niqe_10000_niqe_4.915.pth
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_vi.yml --launcher pytorch --auto_resume


##############################
# stage2: train fusion_segmentation
##############################
# debug
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=4322 basicsr/train_fusion_seg.py -opt options/fusion/train_fusion_seg_MSRS_no_register_debug.yml --launcher pytorch
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=4322 basicsr/train_fusion_seg.py -opt options/fusion/train_fusion_seg_MSRS_no_register_debug.yml --launcher pytorch --auto_resume
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=4322 basicsr/train_fusion_seg.py -opt options/fusion/train_fusion_seg_MSRS_no_register_debug.yml --launcher pytorch 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_fusion_seg.py -opt options/fusion/train_fusion_seg_MSRS_no_register_debug.yml --launcher pytorch
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_fusion_seg.py -opt options/fusion/train_fusion_seg_MSRS_no_register_debug.yml --launcher pytorch --auto_resume
# password="20220926Hyc!@#"
# for i in {1..600}
# do    
#     # echo "$password" | sudo -S ls
#     # sudo pgrep -f python | xargs sudo kill -9
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_fusion_seg.py -opt options/fusion/train_fusion_seg_MSRS_no_register_0406.yml --launcher pytorch --auto_resume        
#     # sleep 30
# done
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_fusion_seg.py -opt options/fusion/train_fusion_seg_MSRS_no_register_0406.yml --launcher pytorch

# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_fusion_seg.py -opt options/fusion/train_fusion_seg_MSRS_no_register.yml --launcher pytorch --auto_resume
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_fusion_seg.py -opt options/fusion/train_fusion_seg_MSRS_no_register.yml --launcher pytorch

##############################
# stage3: train registration
##############################