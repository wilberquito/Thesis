import os
import shutil
import uuid
from collections.abc import Iterable, Iterator, Sized
from pathlib import Path
from typing import List, Union

import cv2
from fastapi import (BackgroundTasks, FastAPI, File, HTTPException, Request,
                     UploadFile)


def save_file_to_disk(file: UploadFile = File(...), save_as="default", folder_name=".") -> Path:
    """
    Save a file into a directory that may exist or not.
    Once the directory is saved return it's content

    Parameters:
    __________
        file: element to save in memory
        save_as: the file's name once serialized in disk
        folder_name: path where the file is saved
    """

    mk_dir(folder_name)
    filename = Path(folder_name) / Path(save_as)
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return filename


def mk_temporal_task(parent_path="./temp"):
    """
    Generates an absolute path to a new task
    """
    task_id = str(uuid.uuid4())
    task_path = Path(parent_path) / Path(task_id)
    return task_path.resolve()


def mk_dir(folder_name: str):
    """
    Makes directory if not exist
    """
    path = Path(folder_name)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def is_file_sanitized(uploaded: UploadFile, supported_content_type=('image/jpeg', 'image/png')) -> bool:
    '''
    Only accepts uploaded files that has content type of jpg or png
    '''
    return uploaded.content_type in supported_content_type


def find_files(parent_dir: Path, extensions: Iterable[str]) -> List[Path]:
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


def read_img(img_path: Path):
    """ Read img from path, transforming to tree channels, rgb """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img