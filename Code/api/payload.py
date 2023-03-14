from pydantic import BaseModel

class IMGC(BaseModel):
    name: str
    base64: str