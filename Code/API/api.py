import os
from typing import List, Union, ValuesView
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
import starlette.status as status
import fastapi
from util import save_file_to_disk, mk_temporal_task, upload_file_sanitized
from vision import mk_prediction

PARENT_PATH_TASK = "./temp"

app = FastAPI()

@app.get("/")
def home(request: Request):
    return fastapi.responses.RedirectResponse('/docs', status_code=status.HTTP_302_FOUND)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
async def predict_single_image(file: UploadFile = File(...)):
    task_path, task_id = mk_temporal_task(parent_path=PARENT_PATH_TASK)
    # content type support
    if not upload_file_sanitized(file):
        raise HTTPException(status_code=400, detail='Content type - %s - not supported' % (file.content_type))
    save_file_to_disk(file, file.filename, task_path)
    predictions = await mk_prediction(task_path)
    return {
        'prediction': predictions
    }

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

