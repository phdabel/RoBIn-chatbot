from pydantic import BaseModel

class RoBInFileInput(BaseModel):
    file_name: str
    content: bytes
    text: str

class RoBInQueryInput(BaseModel):
    text: str

class RoBInQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]