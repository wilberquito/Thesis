from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

import vicorobot.dataset as vd
import vicorobot.utility as vu

from api.dataset import TaskDataset, get_csv
from api.utility import read_yaml

conf = read_yaml('./api.conf.yml')


def __load_wilberquito_model(net_type: str,
                             out_dim: int,
                             pth_path: str,
                             device: str) -> torch.nn.Module:

    raise Exception("Load wilberquito models not implemented yet")


def __load_vicorobot_model(net_type: str,
                           out_dim: int,
                           pth_path: str,
                           device: str) -> torch.nn.Module:

    nn = vu.get_model_class(net_type=net_type)

    pth_file: Path = Path(pth_path)

    if not pth_file.exists():
        raise Exception(f"Vicorobot model - {str(pth_file)} - not found")

    model = nn(enet_type=net_type,
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


def __load_net(model_id: str, device: str) -> Tuple[torch.nn.Module, Dict]:

    meta = get_model_metadata(model_id)
    net_type = meta['net_type']
    out_dim = meta['out_dim']
    pth_path = meta['pth_path']
    mapping = meta['mapping']

    if 'vicorobot' in model_id:
        model = __load_vicorobot_model(net_type=net_type,
                                       out_dim=out_dim,
                                       pth_path=pth_path,
                                       device=device)
    elif 'wilberquito' in model_id:
        model = __load_wilberquito_model(net_type=net_type,
                                         out_dim=out_dim,
                                         pth_path=pth_path,
                                         device=device)
    else:
        raise NotImplementedError()

    return model, mapping


def __load_transforms(model_id: str):

    meta = get_model_metadata(model_id)

    if 'vicorobot' in model_id:
        return vd.get_transforms(image_size=meta['img_size'])
    else:
        raise NotImplementedError()


async def mk_prediction(model_id: str,
                        task_path: Path,
                        save_as: Tuple[str, str]=('classification.csv', 'probabilities.csv')) -> None:

    # Agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Required transformation to make the prediction
    _, val_transforms = __load_transforms(model_id=model_id)

    # Loads the pytorch model
    nn, mapping = __load_net(model_id=model_id, device=device)

    # Create the csv to work with
    csv = get_csv(task_path)

    # Task dataset
    task_dataset = TaskDataset(csv=csv, transform=val_transforms)

    # Task dataloader
    task_dataloader = DataLoader(dataset=task_dataset,
                                 batch_size=len(task_dataset),
                                 shuffle=False)

    # Prediction for each image
    labels = torch.tensor([])
    # Probabilities for all classes
    probabilities = torch.tensor([])
    # Picture names
    names = csv.name

    nn.eval()
    with torch.inference_mode():
         for X in task_dataloader:
            X = X.to(device)
            logits = nn(X)
            probs = torch.softmax(logits, dim=1)
            label = torch.argmax(probs, dim=1)
            labels = torch.cat((labels, label))
            probabilities = torch.cat((probabilities, probs))

    labels = labels.to('cpu').numpy()
    probabilities = probabilities.to('cpu').numpy()

    predictions_csv = pd.DataFrame({
        'name': names,
        'target': labels
    })
    predictions_csv['prediction'] = predictions_csv['target'].map(mapping)

    classes = list(mapping.values())
    probabilities_csv = pd.DataFrame()
    probabilities_csv['name'] = names
    probabilities_csv[classes] = probabilities

    # Saves the classification and the probabilities into 2 separed files
    class_filename, prob_filename = save_as
    predictions_csv.to_csv(task_path / Path(class_filename), index=False)
    probabilities_csv.to_csv(task_path / Path(prob_filename), index=False)


def get_model_metadata(model_id: str) -> dict:

    if not is_model_supported(model_id):
        raise Exception(f'Pytorch model - {model_id} - does not exist')

    return conf['PYTORCH_MODELS'][model_id]


def is_model_supported(model_id: str) -> bool:
    return model_id in get_supported_models()


def get_supported_models() -> List[str]:
    """
    Returns the name of the supported model
    """
    return list(conf['PYTORCH_MODELS'].keys())
