
##############################
# stage1: train LLIE
##############################
# stage 1.1: train_LLIE_vi_NTIRE2024_pretrained
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_vi_NTIRE2024_pretrained.yml --launcher pytorch 

# stage 1.2: train_LLIE_vi_LOL_pretrained
# CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_vi_LOL_pretrained.yml --launcher pytorch 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_vi_LOL_pretrained.yml --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_vi_LOL_pretrained_hyc_small.yml --launcher pytorch

# stage 1.3 train_LLIE_ir
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 --master_port=4322 basicsr/train_LLIE_vi.py -opt options/fusion/train_LLIE_ir.yml --launcher pytorch


##############################
# stage2: train fusion_segmentation
##############################


##############################
# stage3: train registration
##############################