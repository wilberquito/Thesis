from typing import List, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
import os
import shutil
import uuid
from pathlib import Path
from collections.abc import Sized, Iterable, Iterator


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
    Generates a unique identifier for a task and a
    """
    task_id = str(uuid.uuid4())
    task_path = os.path.join(parent_path, task_id)
    return task_path, task_id


def mk_dir(folder_name: str):
    """
    Makes directory if not exist
    """
    path = Path(folder_name)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def upload_file_sanitized(uploaded: UploadFile, supported_content_type=('image/jpeg', 'image/png')) -> bool:
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
