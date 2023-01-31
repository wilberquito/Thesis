import warnings
import torch
from torchvision import models
from PIL import Image
from skimage import io, transform
from pathlib import Path
from util import find_files
import pandas as pd

MODELS = {
    'AlexNet': models.alexnet(pretrained=True)
}


def __get_model(model_name: str = "AlexNet"):
    model = MODELS.get(model_name)
    default_model = models.alexnet(pretrained=True)
    if model is None:
        warnings.warn(
            f"Unknown model {model_name} using default model {default_model._get_name()}))")
        model = default_model

    print(model.__dir__)
    return model


def predict(task_path: str, model_name: str):
    extensions = ('.png', '.jpeg')
    images = find_files(Path(task_path), extensions)
    model = __get_model(model_name)

    for img in images:
        print(img)
