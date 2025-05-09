from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline("text-classification", model="ajaykashela/distilbert_email_classifier_v2")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    result = classifier(input.text)
    return {"result": result}
