from models import Resnest_Melanoma, Seresnext_Melanoma, Effnet_Melanoma
from dataset import get_df
from pathlib import Path
import torch

def get_model_class(net_type='resnet101'):
    if net_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif net_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif 'efficientnet' in net_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()
    return ModelClass

def get_pth_file(parent_dir: Path, eval_type, kernel_type='8c_b3_768_512_18ep', fold=0) -> Path:
    if eval_type == 'best':
        model_file = parent_dir / Path(f'{kernel_type}_best_fold{fold}.pth')
    elif eval_type == 'best_20':
        model_file = parent_dir / Path(f'{kernel_type}_best_20_fold{fold}.pth')
    if eval_type == 'final':
        model_file = parent_dir / Path(f'{kernel_type}_final_fold{fold}.pth')
    else:
        raise NotImplementedError()
    return model_file

if __name__ == "__main__":

    net_type = 'resnet101'
    models_parent_dir = Path('/home/wilber/repos/Thesis/Code/api/pytorch_models')
    eval_type = 'best'
    kernel_type = '8c_b3_768_512_18ep'
    fold = 0

    net_type = 'resnet101'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dim = 8

    params = {
        'models_parent_dir' : Path('/home/wilber/repos/Thesis/Code/api/pytorch_models'),
        'eval_type' : 'best',
        'kernel_type' : '8c_b3_768_512_18ep',
        'fold' : 0
    }

    nn_class = get_model_class(net_type)
    pth_file: Path = get_pth_file(**params)

    model = nn_class(
        enet_type=net_type,
        n_meta_features=0,
        n_meta_dim=[],
        out_dim=out_dim)

    model = model.to(device)

    try:
        model.load_state_dict(torch.load(pth_file), strict=True)
    except Exception as e:
        state_dict = torch.load(pth_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)


    print(model)
