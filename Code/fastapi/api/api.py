from pathlib import Path
from typing import Annotated

import pandas as pd
import starlette.status as status
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile)
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from api.utility import (
    is_uploaded_image_sanitized,
    mk_temporal_task,
    read_yaml,
    save_file_to_disk)
from api.vision import (
    get_supported_models,
    is_model_supported,
    mk_prediction)

env = read_yaml('./conf.yml')

app = FastAPI(title="Melanoma Classifier API Service")


@app.get("/")
def home(request: Request):
    """Redirects to docs url"""
    return RedirectResponse('/docs', status_code=status.HTTP_302_FOUND)


@app.get("/public_models")
async def public_models():
    """
    Returns the name of the available models
    """
    return {
        "models": get_supported_models()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_id='M1'):
    """
    The function receives a file (expected img with jpeg format)
    then it creates the task that is being returned
    and async does the prediction.
    """
    # Sanitize model and img
    __sanitize_model(model_id)
    __sanitize_file(file)

    # New temporal task path
    task_path: Path = mk_temporal_task(parent_path=env['TEMPORAL_TASKS_PATH'])
    task_id: str = task_path.parts[-1]

    # Save image inside the task folder
    save_file_to_disk(parent_dir=task_path,
                      file=file,
                      save_as=str(file.filename))

    # Save and make the prediction into the task directory
    await mk_prediction(model_id=model_id,
                        task_path=task_path)

    return {
        'task_uuid': task_id
    }


@app.post("/predict_bulk")
async def predict_bulk(bg_tasks: BackgroundTasks,
                       files: Annotated[list[UploadFile], File(description="Multiple image files as UploadFile")],
                       model_id='M1'):
    """
    Recives a jar of images and then it creates a task
    for this predict that is returned to consult the result
    of the predictions of each img.
    """

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

    bg_tasks.add_task(mk_prediction,
                      model_id=model_id,
                      task_path=task_path)

    return {
        "task_uuid": task_id,
        "num_images": len(files)
    }


@app.get("/from_task/{task_id}")
async def from_task(task_id: str):
    """
    Consults the predictions from a task
    """
    task_path: Path = Path(env['TEMPORAL_TASKS_PATH']) / Path(task_id)

    __sanitize_path(path=task_path,
                    detail=f'Task - {task_id} - not found')

    classification_filename = env['CLASSIFICATION_SAVE_AS']
    probabilities_filename = env['PROBABILITIES_SAVE_AS']
    about_model_filename = env['ABOUT_MODEL_SAVE_AS']

    class_path = task_path / Path(classification_filename)
    probs_path = task_path / Path(probabilities_filename)
    about_model_path = task_path / Path(about_model_filename)

    for filepath in [class_path, probs_path]:
        error_msg = f'Task - {task_id} - does exists but the prediction is not yet ready.'
        __sanitize_path(path=filepath,
                        detail=error_msg)

    class_csv = pd.read_csv(class_path)
    probs_csv = pd.read_csv(probs_path)
    about_model_dict = pd.read_csv(about_model_path).to_dict('records')[0]

    classification_records = class_csv.to_dict('records')
    response = []

    for record in classification_records:
        record_name = record['name']
        probabilities = probs_csv[probs_csv['name'] == record_name]
        probabilities = probabilities.drop('name', axis=1)

        probabilities_dict = probabilities.to_dict('records')[0]

        resp = {
            'name': record_name,
            'probabilities': probabilities_dict,
            'metadata': about_model_dict,
            'prediction': {
                'target': record['target'],
                'label': record['label'],
                'prediction': record['prediction']
            }
        }
        response.append(resp)

    return response


def __sanitize_model(model_id: str):
    if not is_model_supported(model_id):
        raise HTTPException(status_code=400,
                            detail=f'Pytorch model - {model_id} - not found')


def __sanitize_file(file):
    """
    Check if the file is jpeg or png content type, if not throws and exception
    """

    if not is_uploaded_image_sanitized(file):
        error_msg = f'Content type - {file.content_type} - not supported'
        raise HTTPException(status_code=400, detail=error_msg)


def __sanitize_path(path: Path, detail: str):
    if not path.exists():
        return HTTPException(status_code=500,
                             detail=detail)


origins = env['ALLOW_ORIGINS']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
