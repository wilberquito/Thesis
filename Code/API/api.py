import os
from typing import List, Union, ValuesView
import json

import fastapi
import starlette.status as status
from fastapi import (BackgroundTasks, FastAPI, File, HTTPException, Request,
                     UploadFile)

from vision import mk_prediction
from util import mk_temporal_task, save_file_to_disk, is_file_sanitized, find_files
from pathlib import Path

TMP_PARENT_TASKS = "./temp"

app = FastAPI()

@app.get("/")
def home(request: Request):
    return fastapi.responses.RedirectResponse('/docs', status_code=status.HTTP_302_FOUND)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
async def predict_single_image(model: str, file: UploadFile = File(...)):

    # Is the uploaded file and image?
    __sanitize_file(file)

    # New temporal task path
    task_path = mk_temporal_task(parent_path=TMP_PARENT_TASKS)

    # Save image inside the task folder
    save_file_to_disk(file, file.filename, task_path)

    # Make prediction from task asyncronous
    await mk_prediction(task_path)

    # Returns the unique id of the task generated to consult the prediction late
    return {
        'uuid_task': task_path.parts[-1]
    }

def __sanitize_file(file):
    """
    Check if the file is jpeg or png content type, if not throws and exception
    """
    if not is_file_sanitized(file):
        raise HTTPException(status_code=400, detail='Content type - %s - not supported' % (file.content_type))

@app.post("/predict_pack")
async def predict_images_pack(request: Request, bg_tasks: BackgroundTasks):
    '''
    Function that saves into a unique folder the jar of images from the request.
    So you can consume these images, the uuid of the folder is returned

    Returns
    -------
    task_id: str
        folder where the images where saved

    num_files: int
        number of images saved
    '''
    images = await request.form()
    folder, task_id = mk_temporal_task()
    for image in images.values():
        _ = save_file_to_disk(image, folder_name=folder, save_as=image.filename)
    bg_tasks.add_task(mk_prediction, task_id=task_id)
    return {
        "task_id": task_id,
        "num_files": len(images)
    }


@app.get("/predict_packet_output/{task_id}")
async def predict_images_pack_output(task_id: int):
    '''
    Takes the prediction from task_id folder and returns it.
    May happen that the prediction request and the prediction output
    where faster than the prediction process itself and may not found the prediction,
    in this case, I recommend to consume this end point
    '''
    for file_ in os.listdir(task_id):
        if file_.endswith((".csv")):
            # TODO: transform csv to dict object using DataFrame and return it
            return {
                "task_id": task_id,
                "output": "work in progress"
            }
    return HTTPException(status_code=500, detail='Prediction not found for - %s - task_id' % (task_id))


@app.get("/supported_models")
async def supported_models():
    """
    Description
    ----------
    returns the name of the available models to make prediction of skin cancers
    """
    return {
        "support": vision.list_models()
    }