import warnings
import torch
from torchvision import models
from PIL import Image
from skimage import io, transform
from pathlib import Path
from util import find_files
import pandas as pd

DEFAULT_MODEL = 'AlexNet'

MODELS = {
    'AlexNet': models.alexnet(pretrained=True)
}

def __get_model(model_name: str):
    model = MODELS.get(model_name)
    default_model = models.alexnet(pretrained=True)
    if model is None:
        warnings.warn(
            f"Unknown model {model_name} using default model {default_model._get_name()}))")
        model = default_model

    print(model.__dir__)
    return model


async def mk_prediction(task_path: str, model_id: str = DEFAULT_MODEL):
    extensions = ('.png', '.jpeg')
    images_path = find_files(Path(task_path), extensions)
    images_id = list(map(lambda x : x.parts[-1], images_path))
    model = __get_model(model_id)

    return images_id




