import os
import shutil
from typing import List, Union, ValuesView
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks 
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

app = FastAPI()

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
async def predict_single_image(file: UploadFile = File(...)):
    folder, task_id = _mk_temporal_task()
    # content type support
    if not _is_image_sanitized(file):
        raise HTTPException(status_code=400, detail='Content type - %s - not supported' % (file.content_type))
    _save_file_to_disk(file, folder, file.filename)
    predictions = _make_predictions(task_id)
    return {
        'prediction': predictions 
    }

async def _make_predictions(task_id, model='cnn') -> List[str]:
    # TODO: call prediction, for now it returns the name of how it was saved
    return ["p1", "p2"]
    
def _save_file_to_disk(file: UploadFile = File(...), folder_name=".", save_as="default") -> str:
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    temp_file = os.path.join(folder_name, save_as)
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return temp_file

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
    folder, task_id = _mk_temporal_task()
    for image in images.values():
        _ = _save_file_to_disk(image, folder_name=folder, save_as=image.filename)
    bg_tasks.add_task(_make_predictions, task_id=task_id)
    return {
        "task_id": task_id,
        "num_files": len(images)
    }
    
def _mk_temporal_task():
    task_id = str(uuid.uuid4())
    folder = os.path.join("temp", task_id)
    os.mkdir(folder)
    return folder, task_id

def _is_image_sanitized(uploaded: UploadFile | str) -> bool:
    '''
    Only accepts uploaded files that has content type of jpg or png
    '''
    if not isinstance(uploaded, UploadFile):
        return False 
    supported_content_type = ('image/jpeg', 'image/png')
    return uploaded.content_type in supported_content_type

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

