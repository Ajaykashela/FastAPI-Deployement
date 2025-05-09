from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline(
    "fill-mask",  # distilbert-base-uncased is a masked language model
    model="distilbert-base-uncased"
)

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    result = classifier(input.text)
    return {"result": result}
