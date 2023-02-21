import logging
import warnings
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from PIL import Image
from skimage import io, transform
from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torchvision.transforms as transforms

from util import find_files

# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset

# 1. Subclass torch.utils.data.Dataset

DEFAULT_MODELS_PARENT_DIR = './models'
DEFAULT_MODEL = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
SUPPORTED_MODELS = {}


def __load_models_from_disk():
    """
    Description
    __________
    Search recursively in
    """
    # parent_path = Path(DEFAULT_MODELS_PARENT_DIR)
    # for path in parent_path:
    #     model_id = path.parts[-1].split('.')[-2]
    #     SUPPORTED_MODELS[model_id] = torch.load(path)
    pass


def __get_model(model_name: str):
    model = SUPPORTED_MODELS.get(model_name)
    if model is None:
        logging.warn(
            f"Unknown model {model_name} using default model {DEFAULT_MODEL._get_name()}))")
        model = DEFAULT_MODEL
    return model


async def mk_prediction(targ_dir: Path):
    paths = find_files(targ_dir, ('.png', '.jpeg')) # note: you'd have to update this if you've got .png's or .jpeg's
    images = [Image.open(img).convert('RGB') for img in paths]
    prediction = [__default_prediction(img) for img in images]
    export_prediction_to_csv(targ_dir / Path('prediction.csv'), paths, prediction)

def export_prediction_to_csv(csv_name, images_paths, predictions):
    pass

def __pil_to_tensor(pil):
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    return transform(pil)


def __default_prediction(img: Image.Image):
    from torchvision.io import read_image
    from torchvision.models import resnet50, ResNet50_Weights

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")
    return {
        'category_id': class_id,
        'category_name': category_name,
        'score': score
    }

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
