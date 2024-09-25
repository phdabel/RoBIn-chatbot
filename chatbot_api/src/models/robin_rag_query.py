from pydantic import BaseModel

class RoBInQueryInput(BaseModel):
    text: str

class RoBInQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]