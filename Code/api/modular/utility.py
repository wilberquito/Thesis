import os
import shutil
import uuid
from collections.abc import Iterable, Iterator, Sized
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import yaml
from fastapi import (BackgroundTasks, FastAPI, File, HTTPException, Request,
                     UploadFile)
from PIL import Image


def save_file_to_disk(parent_dir: Path,
                      file: UploadFile = File(...),
                      save_as: str ="default") -> Path:
    """
    Save a file into a directory that may exist or not.
    Once the directory is saved return it's content

    Parameters:
    __________
        file: element to save in memory
        save_as: the file's name once serialized in disk
        folder_name: path where the file is saved
    """

    mk_dir(parent_dir)
    filename = Path(parent_dir) / Path(save_as)
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return filename


def mk_temporal_task(parent_path="./temp") -> Path:
    """
    Generates an absolute path to a new task
    """
    task_id = str(uuid.uuid4())
    task_path = Path(parent_path) / Path(task_id)
    return task_path.resolve()


def mk_dir(folder_name: Union[str,Path]):
    """
    Makes directory if not exist
    """
    path = Path(folder_name)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def is_uploaded_image_sanitized(uploaded: UploadFile,
                      supported_content_type=('image/jpeg', 'image/png')) -> bool:
    '''
    Only accepts uploaded files that has content type of jpg or png
    '''
    check_content_type = uploaded.content_type in supported_content_type
    check_name = uploaded.filename is not None

    return all([
                   check_content_type,
                   check_name
               ])


def find_files(parent_dir: Path, extensions: list[str]) -> List[Path]:
    """
    Description
    -----------

    Grep all files from the directory that matches any of the extensions.
    It's a recursive process.

    Parameters
    ----------

    parent_dir: Path

    root path to find the files that ends with any of the extension

    extensions: Iterable[str]

    usuful to pick the files that matches with any of this strings
    """
    matches = []
    for root, _, files in os.walk(parent_dir):
        for file in files:
            check = [file.endswith(e) for e in extensions]
            if any(check):
                matches.append(Path(os.path.join(root, file)))
    return matches


def path_to_tensor(img_path: Path, transform=None) -> torch.Tensor:
    # Load img as PIL object
    pil_img = Image(img_path)

    if transform is not None:
        res = transform(image=pil_img)
        img = res['image'].astype(np.float32)
    else:
        img = img.astype(np.float32)

    # Channel first
    img = img.transpose(2, 0, 1)

    # Convert PIL img to Pytorch tensor
    tensor =  torch.Tensor(img)

    return tensor


def read_yaml(file_path: Path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)