from typing import List, Union, ValuesView
import pandas as pd

import fastapi
import starlette.status as status
from fastapi import (BackgroundTasks, FastAPI, File, HTTPException, Request,
                     UploadFile)

from modular.vision import mk_prediction, get_supported_models, is_model_supported
from modular.utility import mk_temporal_task, save_file_to_disk, is_uploaded_image_sanitized, read_yaml

from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

conf = read_yaml('./api.conf.yml')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)


@app.get("/")
def home(request: Request):
    return fastapi.responses.RedirectResponse('/docs', status_code=status.HTTP_302_FOUND)


@app.get("/supported_models")
async def supported_models():
    """
    Description
    ----------
    returns the name of the available models to make prediction of skin cancers
    """
    return {
        "models": get_supported_models()
    }


@app.get("/from_task/{task_id}")
async def from_task(task_id: str):
    '''
    Consult a task resulting predictions
    '''

    task_path: Path = Path(conf['TEMPORAL_TASKS_PATH']) / Path(task_id)

    __sanitize_path(path=task_path,
                    detail=f'Task - {task_id} - not found')

    predict_path = task_path / Path(conf['PREDICTION_SAVE_AS'])

    __sanitize_path(path=predict_path,
                    detail=f'Task - {task_id} - does exists but the prediction is not yet ready. Try it latter')

    csv: pd.DataFrame = pd.read_csv(predict_path)
    csv: dict = csv.to_dict('records')
    return csv


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_id='vicorobot.8c_b3_768_512_18ep_best_fold0'):

    # Check if the Pytorch model is available
    __sanitize_model(model_id)

    # Is the uploaded file and image?
    __sanitize_file(file)

    # New temporal task path
    task_path: Path = mk_temporal_task(parent_path=conf['TEMPORAL_TASKS_PATH'])
    task_id: str = task_path.parts[-1]

    # Save image inside the task folder
    save_file_to_disk(parent_dir=task_path,
                      file=file,
                      save_as=str(file.filename))


    # Save and make the prediction into the task directory
    await mk_prediction(model_id=model_id,
                        task_id=task_path,
                        save_as=conf['PREDICTION_SAVE_AS'])

    return {
        'uuid_task': task_id
    }


@app.post("/predict_bulk")
async def predict_bulk(request: Request,
                       bg_tasks: BackgroundTasks,
                       model_id='vicorobot.efficientnet_b3'):

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
    # Takes the files from the form request
    files = await request.form()

    # Creates a new task
    task_id: Path = mk_temporal_task(parent_path=conf['TEMPORAL_TASKS_PATH'])

    # Sanitize each image and save inside the task
    for file in files.values():

        # Check the state of the file
        __sanitize_file(file)

        # Save image inside the task folder
        save_file_to_disk(parent_dir=task_id,
                          file=file,
                          save_as=str(file.filename))

    bg_tasks.add_task(mk_prediction,
                      model_id=model_id,
                      task_id=task_id,
                      save_as=conf['PREDICTION_SAVE_AS'])

    return {
        "task_id": task_id,
        "num_files": len(files)
    }


def __sanitize_model(model_id: str):
    if not is_model_supported(model_id):
        raise HTTPException(status_code=400,
                            detail=f'Pytorch model - {model_id} - not found')

def __sanitize_file(file):
    """
    Check if the file is jpeg or png content type, if not throws and exception
    """
    if not is_uploaded_image_sanitized(file):
        raise HTTPException(status_code=400, detail='Content type - %s - not supported' % (file.content_type))


def __sanitize_path(path: Path, detail: str):

    if not path.exists():
        return HTTPException(status_code=500,
                             detail=detail)
