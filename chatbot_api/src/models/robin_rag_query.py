from pydantic import BaseModel
from fastapi import UploadFile, File

class RoBInFileInput(BaseModel):
    uploaded_file: UploadFile = File(...)
    query_text: str

class RoBInQueryInput(BaseModel):
    text: str
    session: str

class RoBInQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]

class RoBInFileOutput(BaseModel):
    query_text: str
    answer: dict