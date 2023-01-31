from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
import os
import shutil
import uuid


def save_file_to_disk(file: UploadFile = File(...), save_as="default", folder_name=".") -> str:
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
    temp_file = os.path.join(folder_name, save_as)
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return temp_file


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
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def upload_file_sanitized(uploaded: UploadFile, supported_content_type=('image/jpeg', 'image/png')) -> bool:
    '''
    Only accepts uploaded files that has content type of jpg or png
    '''
    return uploaded.content_type in supported_content_type
