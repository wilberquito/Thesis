import logging
import warnings
from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image
from skimage import io, transform
from torchvision.models import resnet50, ResNet50_Weights

from util import find_files

DEFAULT_MODELS_PARENT_DIR = './models'
DEFAULT_MODEL = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
SUPPORTED_MODELS = {}


def __load_models_from_disk():
    """
    Description
    __________
    Search recursively in
    """
    parent_path = Path(DEFAULT_MODELS_PARENT_DIR)
    for path in parent_path:
        model_id = path.parts[-1].split('.')[-2]
        SUPPORTED_MODELS[model_id] = torch.load(path)


def __get_model(model_name: str):
    model = SUPPORTED_MODELS.get(model_name)
    if model is None:
        logging.warn(
            f"Unknown model {model_name} using default model {DEFAULT_MODEL._get_name()}))")
        model = DEFAULT_MODEL
    return model


async def mk_prediction(task_path: str, model_id: str = DEFAULT_MODEL):
    extensions = ('.png', '.jpeg')
    images_path = find_files(Path(task_path), extensions)
    images_id = list(map(lambda x: x.parts[-1], images_path))
    model = __get_model(model_id)

    return images_id


def list_models() -> List[str]:
    """
    Description
    -----------
    Returns the name of the supported model
    """
    return list(SUPPORTED_MODELS.keys())


def run():
    logging.info("Loading models from disk...")
    __load_models_from_disk()
    logging.info(f"Models supported: {list_models()}")
