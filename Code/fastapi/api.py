from pathlib import Path
from typing import Annotated, cast, Dict

import fastapi
import pandas as pd
import starlette.status as status
from fastapi import (BackgroundTasks,
                     FastAPI,
                     File,
                     HTTPException,
                     Request,
                     UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi_versioning import VersionedFastAPI, version

from api.utility import (is_uploaded_image_sanitized,
                         mk_temporal_task,
                         read_yaml,
                         save_file_to_disk)
from api.vision import (get_supported_models,
                        is_model_supported,
                        mk_prediction)

env = read_yaml(Path('env.yml'))
KEY_PROBS = 'PROBABILITIES_SAVE_AS'
KEY_CLASS = 'CLASSIFICATION_SAVE_AS'

app = FastAPI(title="SIIM-ISIC Melanoma Pytorch service")


@app.get("/")
def home(_: Request):
    return fastapi.responses. \
        RedirectResponse('/docs', status_code=status.HTTP_302_FOUND)


@app.get("/public_models")
@version(1, 0)
async def public_models():
    """
    Description
    ----------
    returns the name of the available models to make prediction of skin cancers
    """
    return {
        "models": get_supported_models()
    }


@app.get("/from_task/{task_id}")
@version(1, 0)
async def from_task(task_id: str):
    '''
    Consult a task resulting predictions
    '''

    task_path: Path = Path(env['TEMPORAL_TASKS_PATH']) / Path(task_id)

    __sanitize_path(path=task_path,
                    detail=f'Task - {task_id} - not found')

    class_filename, probs_filename = (env[KEY_CLASS], env[KEY_PROBS])
    class_path = task_path / Path(class_filename)
    probs_path = task_path / Path(probs_filename)

    for filepath in [class_path, probs_path]:
        __sanitize_path(path=filepath,
                        detail=f'Task - {task_id} - does exists but the prediction is not yet ready. Try it latter')

    class_csv = cast(pd.DataFrame, pd.read_csv(class_path))
    probs_csv = cast(pd.DataFrame, pd.read_csv(probs_path))

    response = class_csv.to_dict('records')
    for resp in response:
        name = resp['name']
        probs = cast(pd.DataFrame, probs_csv[probs_csv['name'] == name])
        probs = probs.drop('name', axis=1)
        resp['probs'] = cast(Dict, probs.to_dict('records')[0])

    return response


@app.post("/predict")
@version(1, 0)
async def predict(file: UploadFile = File(...), model_id='vicorobot.8c_b3_768_512_18ep_best_fold0'): # Check if the Pytorch model is available
    __sanitize_model(model_id)

    # Is the uploaded file and image?
    __sanitize_file(file)

    # New temporal task path
    task_path: Path = mk_temporal_task(parent_path=env['TEMPORAL_TASKS_PATH'])
    task_id: str = task_path.parts[-1]

    # Save image inside the task folder
    save_file_to_disk(parent_dir=task_path,
                      file=file,
                      save_as=str(file.filename))

    save_as = (env[KEY_CLASS], env[KEY_PROBS])
    # Save and make the prediction into the task directory
    await mk_prediction(model_id=model_id,
                        task_path=task_path,
                        save_as=save_as)

    return {
        'task_uuid': task_id
    }


@app.post("/predict_bulk")
@version(1, 0)
async def predict_bulk(bg_tasks: BackgroundTasks,
                       files: Annotated[list[UploadFile], File(description="Multiple image files as UploadFile")],
                       model_id='vicorobot.8c_b3_768_512_18ep_best_fold0'):

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

    # Check if the Pytorch model is available
    __sanitize_model(model_id)

    # Creates a new task
    task_path: Path = mk_temporal_task(parent_path=env['TEMPORAL_TASKS_PATH'])
    task_id: str = task_path.parts[-1]

    # Sanitize each image and save inside the task
    for file in files:

        # Check the state of the file
        __sanitize_file(file)

        # Save image inside the task folder
        save_file_to_disk(parent_dir=task_path,
                          file=file,
                          save_as=str(file.filename))

    save_as = (env[KEY_CLASS], env[KEY_PROBS])
    bg_tasks.add_task(mk_prediction,
                      model_id=model_id,
                      task_path=task_path,
                      save_as=save_as)

    return {
        "task_uuid": task_id,
        "num_images": len(files)
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


app = VersionedFastAPI(app, default_api_version=(1, 0))

origins = env['ALLOW_ORIGINS']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
