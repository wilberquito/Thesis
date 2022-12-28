import os
import shutil
from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException

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
        'tempFilename': temp_file 
    }
    
def _save_file_to_disk(file: UploadFile = File(...), path=".", save_as="default"):
    if not os.path.exists(path):
        os.makedirs(path)
    extension = os.path.splitext(file.filename)[-1]
    temp_file = os.path.join(path, save_as + extension)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_file

@app.get("/ml-models")
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