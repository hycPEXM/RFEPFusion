import torch

from basicsr.archs.fusion.LLIE_arch import LLIE_VI, LLIE_IR
from basicsr.archs.fusion.net_utils import init_weights
# from basicsr.archs.fusion.RFEPFusion_arch import RFEPFusion_no_register, RFEPFusion
# from basicsr.archs.fusion.mscan import MSCAN

def init_pretrained_LLIE(modality = 'vi', 
                         pretrained_path = "/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/segnext_small_1024x1024_city_160k.pth", 
                         prefix_unmatched = ('backbone.', 'vi_encoder.'),
                         save_path = "/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/LLIE_vi_init.pth"):
    if modality == 'vi':
        model = LLIE_VI()
    elif modality == 'ir':
        model = LLIE_IR()
    else:
        raise ValueError(f"不支持的模态: {modality}, 只支持 'vi' 或 'ir'")
    model.apply(init_weights)
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    # model_dict = model.state_dict()
    # for key in pretrained_dict['state_dict'].keys():
    #     print(key)
    # for key in model_dict.keys():
    #     print(key)
    
    # replace key with unmatched prefix 
    pretrained_dict = {k.replace(prefix_unmatched[0], prefix_unmatched[1]): v \
        for k,v in pretrained_dict['state_dict'].items() if k.startswith(prefix_unmatched[0])}
    model.load_state_dict(pretrained_dict, strict=False)
    # Don not use EMA model
    torch.save({'params': model.state_dict()}, save_path)
    
def init_pretrained_RFEPFusion_no_register():
    pass

def init_pretrained_RFEPFusion():
    pass


if __name__ == "__main__":
    init_pretrained_LLIE()
    init_pretrained_LLIE(modality='ir', 
                         pretrained_path="/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/segnext_tiny_1024x1024_city_160k.pth", 
                         prefix_unmatched=('backbone.', 'ir_encoder.'),
                         save_path="/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/LLIE_ir_init.pth")
