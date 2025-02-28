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

def remove_keys_with_prefix(state_dict, prefix):
    """
    Remove all key-value pairs from the state_dict where the key starts with the given prefix.

    Args:
        state_dict (dict): The model's state_dict.
        prefix (str): The prefix to match for removal.

    Returns:
        dict: The modified state_dict.
    """
    keys_to_remove = [key for key in state_dict.keys() if key.startswith(prefix)]
    for key in keys_to_remove:
        del state_dict[key]
    return state_dict

def init_LLIE_ir_only_encoder():
    ir_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/RFEPFusion_LLIE_ir/best_psnr_66000_psnr_47.171_ssim_0.992.pth'
    model = torch.load(ir_path)
    save_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/ir_encoder.pth'
    torch.save({'params': remove_keys_with_prefix(model['params'], 'ir_decoder.')}, save_path)

def init_LLIE_vi_only_encoder():
    vi_path = ''
    model = torch.load(vi_path)
    save_path = '/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/vi_encoder.pth'
    torch.save({'params': remove_keys_with_prefix(model['params'], 'vi_decoder.')}, save_path)

def init_pretrained_RFEPFusion_no_register():
    pass

def init_pretrained_RFEPFusion():
    pass


if __name__ == "__main__":
    # init_pretrained_LLIE()
    # init_pretrained_LLIE(modality='ir', 
    #                      pretrained_path="/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/segnext_tiny_1024x1024_city_160k.pth", 
    #                      prefix_unmatched=('backbone.', 'ir_encoder.'),
    #                      save_path="/home/hongyuchen/master_thesis/RFEPFusion/experiments/pretrained_models/LLIE_ir_init.pth")
    init_LLIE_ir_only_encoder()