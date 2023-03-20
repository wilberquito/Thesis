from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

import vicorobot as vi

from .dataset import TaskDataset, get_csv

# 1. Subclass torch.utils.data.Dataset

PATH_PYTORCH_MODELS = Path('/home/wilberquito/pytorch/trained/melanoma')
PYTORCH_MODELS = {
    'vicorobot.efficientnet_b3': {
        'net_type': 'efficientnet_b3',
        'image_size': 320,
        'pth_file': {
            'parent_dir': PATH_PYTORCH_MODELS / Path('vicorobot'),
            'eval_type': 'best',
            'out_dim': 8,
            'kernel_type': '8c_b3_768_512_18ep',
            'fold': 0
        }

    }
}

def __load_vicorobot_model(device: str,
                           net_type: str,
                           parent_dir: Path,
                           eval_type: str = 'best',
                           out_dim: int = 8,
                           kernel_type: str = '8c_b3_768_512_18ep',
                           fold: int=0) -> torch.nn.Module:
    nn = vi.utility.get_model_class(net_type=net_type)
    pth_file = vi.utility.get_path_file(parent_dir=parent_dir,
                                        eval_type=eval_type,
                                        kernel_type=kernel_type,
                                        fold=fold)

    model = nn(
        enet_type=net_type,
        n_meta_features=0,
        n_meta_dim=[],
        out_dim=out_dim)

    model = model.to(device)

    try:
        model.load_state_dict(torch.load(pth_file,
                                         map_location=device),
                              strict=True)
    except Exception as e:
        state_dict = torch.load(pth_file,
                                map_location=device)

        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)

    return model


def __mk_net(device: str,
             net_id: str) -> torch.nn.Module:

    meta_model = PYTORCH_MODELS[net_id]

    if 'vicorobot' in net_id:
        model = __load_vicorobot_model(device=device,
                                       net_type=meta_model['net_type'],
                                       **meta_model['pth_file'])
    else:
        raise NotImplementedError()

    return model

def get_transforms(model_id: str, image_size: int):

    if 'vicorobot' in model_id:
        return vi.dataset.get_transforms
    else:
        raise NotImplementedError()


async def mk_prediction(model_id: str,
                        task_id: Path,
                        save_as='task_prediction.csv') -> None:

    # Agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model metadata
    metadata = PYTORCH_MODELS[model_id]

    # Required transformation to make the prediction
    _, val_transforms = get_transforms(model_id=model_id,
                                       image_size=metadata['image_size'])

    # Loads the pytorch model
    nn = __mk_net(model_id)

    # Create the csv to work with
    csv = get_csv(task_id)

    # Task dataset
    task_dataset = TaskDataset(csv=csv, transform=val_transforms)

    # Task dataloader
    task_dataloader = DataLoader(dataset=task_dataset,
                                 batch_size=8,
                                 shuffle=False)

    predictions = []
    names = csv.name

    with torch.inference_mode():
         for X in task_dataloader:
            X = X.to(device)
            logits = nn(X)
            pred = torch.argmax(torch.softmax(logits))
            predictions.append(pred)

    predictions_csv = pd.DataFrame({
        'name': names,
        'prediction': predictions
    })

    predictions_csv.to_csv(task_id / Path(save_as))


def get_supported_models() -> List[str]:
    """
    Returns the name of the supported model
    """
    global NAMES_PYTORCH_MODELS
    return NAMES_PYTORCH_MODELS.keys()

