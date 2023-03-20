from pathlib import Path

from .models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma


def get_model_class(net_type='resnest101'):
    if net_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif net_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif net_type == 'efficientnet_b3':
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()
    return ModelClass


def get_pth_file(parent_dir: Path,
                 eval_type,
                 kernel_type='8c_b3_768_512_18ep',
                 fold=0) -> Path:
    if eval_type == 'best':
        model_file = parent_dir / Path(f'{kernel_type}_best_fold{fold}.pth')
    elif eval_type == 'best_20':
        model_file = parent_dir / Path(f'{kernel_type}_best_20_fold{fold}.pth')
    elif eval_type == 'final':
        model_file = parent_dir / Path(f'{kernel_type}_final_fold{fold}.pth')
    else:
        raise NotImplementedError()
    return model_file
