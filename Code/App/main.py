import os
import shutil
from typing import Union
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks 

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
async def perform_predict(file: UploadFile = File(...),):
    # content type support
    content_type_supported = ('image/jpeg', 'image/png')
    if file.content_type not in content_type_supported:
        raise HTTPException(status_code=400, detail='Content type - %s - not supported' % (file.content_type))
    temp_file = _save_file_to_disk(file, path="temp", save_as="latest")
    return {
        'filename': file.filename, 
        'file_id': temp_file
    }
    
def _save_file_to_disk(file: UploadFile = File(...), path=".", save_as="default"):
    if not os.path.exists(path):
        os.makedirs(path)
    extension = os.path.splitext(file.filename)[-1]
    temp_file = os.path.join(path, save_as + extension)
    with open(temp_file, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return temp_file

@app.post("/predict_packet")
async def perform_predict_packet(request: Request, bg_tasks: BackgroundTasks):
    """
    It avoids users to await to the prediction of every image. Instead, once the
    images are downloaded I send the users the uuid of the folder where those images where saved.

    docs: https://fastapi.tiangolo.com/tutorial/background-tasks/?h=background
    """
    images = await request.form()
    folder_name = str(uuid.uuid4())
    os.mkdir(folder_name)

    for image in images.values():
        _ = _save_file_to_disk(image, path=folder_name, save_as=image.filename)

    bg_tasks.add_task(read_images_from_disk, folder_name)
    return {
        "task_id": folder_name,
        "num_files": len(images)
    }
    
@app.get("/ml_models")
async def get_models():
    allowed_extensions = ('.h5', '.txt')
    models = list_files('models', allowed_extensions)
    return {'support': models }

def list_files(top, allowed_extensions=()):
    for _, _, files in os.walk(top, topdown=False):
        return [name for name in files if name.endswith(allowed_extensions)]

async def load_model(filename = 'h5.txt'):
    assert filename.endswith('.txt'), 'Only txt models are allowed'
    with open(filename, 'r') as f:
        content = f.read()
    return content

async def read_images_from_disk(folder_name):
    print(folder_name)
    pass