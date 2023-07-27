from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import vicorobot.dataset as vd
import modular.dataset as md

from api.dataset import TaskDataset, get_csv
from api.utility import read_yaml, get_model_class

env = read_yaml('./conf.yml')


def load_net(model_id: str, device: str) -> Tuple[torch.nn.Module, Dict]:

    meta = get_model_metadata(model_id)
    origin = meta['origin']
    net_type = meta['net_type']
    out_dim = meta['out_dim']
    pth_path = meta['pth_path']
    mapping = meta['mapping']
    class_name = get_model_class(origin, net_type)

    if origin == 'vicorobot':
        model = __load_vicorobot_model(class_nn=class_name,
                                       net_type=net_type,
                                       out_dim=out_dim,
                                       pth_path=pth_path,
                                       device=device)
    elif origin == 'wilberquito':
        model = __load_wilberquito_model(class_nn=class_name,
                                         out_dim=out_dim,
                                         pth_path=pth_path,
                                         device=device)
    else:
        raise NotImplementedError()

    return model, mapping


def __load_wilberquito_model(class_nn: torch.nn.Module,
                             out_dim: int,
                             pth_path: str,
                             device: str) -> torch.nn.Module:

    pytorch_model_path = Path(pth_path)
    if not pytorch_model_path.exists():
        exc_msg = f"Vicorobot model - {str(pytorch_model_path)} - not found"
        raise Exception(exc_msg)

    instance_nn = class_nn(out_dim)
    instance_nn = instance_nn.to(device)
    checkpoint = torch.load(pytorch_model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    instance_nn.load_state_dict(model_state_dict,
                                strict=True)
    return instance_nn


def __load_vicorobot_model(class_nn: torch.nn.Module,
                           net_type: str,
                           out_dim: int,
                           pth_path: str,
                           device: str) -> torch.nn.Module:

    pth_file = Path(pth_path)

    if not pth_file.exists():
        raise Exception(f"Vicorobot model - {str(pth_file)} - not found")

    instance_nn = class_nn(enet_type=net_type,
                           n_meta_features=0,
                           n_meta_dim=[],
                           out_dim=out_dim)

    instance_nn = instance_nn.to(device)

    try:
        instance_nn. \
            load_state_dict(torch.load(pth_file, map_location=device),
                            strict=True)
    except Exception as e:
        state_dict = torch.load(pth_file,
                                map_location=device)

        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        instance_nn.load_state_dict(state_dict, strict=True)

    return instance_nn


def __load_transforms(model_id: str):

    meta = get_model_metadata(model_id)
    img_size = meta['img_size']
    origin = meta['origin']

    if 'vicorobot' == origin:
        return vd.get_transforms(image_size=meta['img_size'])
    elif 'wilberquito' == origin:
        return md.get_transforms(image_size=img_size)
    else:
        raise NotImplementedError()


async def mk_prediction(model_id: str,
                        task_path: Path) -> None:

    # Agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Required transformation to make the prediction
    _, val_transforms = __load_transforms(model_id=model_id)

    # Loads the pytorch model
    nn, mapping = load_net(model_id=model_id, device=device)

    # Create the csv to work with
    csv = get_csv(task_path)

    # Task dataset
    task_dataset = TaskDataset(csv=csv, transform=val_transforms)

    # Task dataloader
    task_dataloader = DataLoader(dataset=task_dataset,
                                 batch_size=len(task_dataset),
                                 shuffle=False)

    # Prediction for each image
    labels = torch.tensor([], device=device)
    # Probabilities for all classes
    probabilities = torch.tensor([], device=device)
    # Picture names
    names = csv.name

    nn.eval()
    with torch.inference_mode():
        for X in task_dataloader:
            X = X.to(device)
            logits = nn(X)
            probs = torch.softmax(logits, dim=1)
            label = torch.argmax(probs, dim=1)

            # Saving the labels and probabilities
            labels = torch.cat((labels, label))
            probabilities = torch.cat((probabilities, probs))

    labels = labels.to('cpu').numpy()
    probabilities = probabilities.to('cpu').numpy()
    targets = np.full(len(labels),
                      target_model(model_id))
    targets = targets == labels

    predictions_csv = pd.DataFrame({
        'name': names,
        'label': labels,
        'target': targets
    })
    predictions_csv['prediction'] = predictions_csv['label'].map(mapping)

    classes = list(mapping.values())
    probabilities_csv = pd.DataFrame()
    probabilities_csv['name'] = names
    probabilities_csv[classes] = probabilities

    about_model_data = about_model(model_id)
    about_model_csv = pd.DataFrame(columns=about_model_data.keys())
    about_model_csv.loc[len(about_model_csv.index)] = about_model_data.values()

    # Saves metada about the predictions and model used in the predictions
    classification_filename = env['CLASSIFICATION_SAVE_AS']
    probabilities_filename = env['PROBABILITIES_SAVE_AS']
    about_model_filename = env['ABOUT_MODEL_SAVE_AS']

    predictions_csv.to_csv(task_path / Path(classification_filename),
                           index=False)
    probabilities_csv.to_csv(task_path / Path(probabilities_filename),
                             index=False)
    about_model_csv.to_csv(task_path / Path(about_model_filename), index=False)


def target_model(model_id: str) -> int:
    metadata = get_model_metadata(model_id)
    return metadata['target']


def about_model(model_id: str) -> dict:
    """Returns especific information about the model"""
    metadata = get_model_metadata(model_id)
    keywords = ['origin', 'net_type', 'img_size', 'out_dim']
    about = {k: v for k, v in metadata.items() if k in keywords}
    return about


def get_model_metadata(model_id: str) -> dict:
    """Returns hole information about the model"""

    if not is_model_supported(model_id):
        raise Exception(f'Pytorch model - {model_id} - does not exist')

    return env['PYTORCH_MODELS'][model_id]


def is_model_supported(model_id: str) -> bool:
    return model_id in get_supported_models()


def get_supported_models() -> List[str]:
    """
    Returns the name of the supported model
    """
    return list(env['PYTORCH_MODELS'].keys())
