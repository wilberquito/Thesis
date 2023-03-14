import logging
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

from api.utility import find_files, read_img
from nn.dataset import MelanomaDataset, get_transforms
from nn.models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma

# 1. Subclass torch.utils.data.Dataset

DEFAULT_MODELS_PARENT_DIR = './models'
DEFAULT_MODEL = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
SUPPORTED_MODELS = {}
IMG_SIZE = 576
device = 'cpu'


def __mk_net(net_type, out_dim, pretrained) -> torch.nn.Module:
    if net_type == 'resnest101':
        net = Resnest_Melanoma(net_type, out_dim, pretrained)
    elif net_type == 'seresnext101':
        net = Seresnext_Melanoma(net_type, out_dim, pretrained)
    elif 'efficientnet' in net_type:
        net = Effnet_Melanoma(net_type, out_dim, pretrained)
    else:
        raise NotImplementedError()
    return net

def __load_models_from_disk():
    """
    Description
    -----------
    Finds files with .pth and .pt extension in subfolder <DEFAULT_MODEL_PARENT_DIR>
    """
    paths = find_files(DEFAULT_MODELS_PARENT_DIR, ('.pth'))
    models = []
    for path in paths:
        # Take the filename model from the path
        net_type = path.parts[-1].split('.')[0]
        # Save path to the model (k, v)
        SUPPORTED_MODELS[net_type] = path


async def mk_prediction(net_type, targ_dir: Path, save_as='task_prediction.csv'):
    paths = find_files(targ_dir, ('.png', '.jpeg')) # note: you'd have to update this if you've got .png's or .jpeg's
    _, transforms_val = get_transforms(IMG_SIZE)
    csv = pd.DataFrame({ 'filepath': paths })
    dataset = MelanomaDataset(csv, 'test', transforms_val)
    dataloader = DataLoader(dataset)

    y_pred_class = []

    net = __mk_net(net_type, 8, True)
    net = net.to(device)
    net.eval()
    with torch.inference_mode():
        for X in dataloader:
            X = X.to(device)
            logits = net(X)
            pred = torch.argmax(torch.softmax(logits))
            y_pred_class.append(pred)

    task_prediction = pd.DataFrame({
        'filepath': csv.filepath,
        'prediction': y_pred_class
    })

    save_path = targ_dir / Path(save_as)
    task_prediction.to_csv(save_path)


def get_supported_models() -> List[str]:
    """
    Returns the name of the supported model
    """
    return list(SUPPORTED_MODELS.keys())


def run():
    logging.info("Loading models from disk...")
    __load_models_from_disk()
    logging.info(f"Models loaded : {get_supported_models()}")
    logging.info('Selecting device to work with')
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device selected to work with: {device}')
