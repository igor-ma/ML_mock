'''Module for registering API templates to use for deploy'''

class APITemplates:
    '''Class to save API templates'''

    def __init__(self) -> None:
        self.buildTemplate1()
    
    def getFilledTemplate1(self, param1, param2):
        return self.template_1.format(param1, param2)
    
    def buildTemplate1(self):
        self.template_1 = """
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
import torch


app = FastAPI()

#load model
tokenizer = AutoTokenizer.from_pretrained('{}')
model = AutoModelForSequenceClassification.from_pretrained('{}')

#request data type
class PredictionRequest(BaseModel):
    text: str

@app.post('/predict')
def predict(request_data: PredictionRequest):
    text = request_data.text
    
    #tokenize text
    inputs = tokenizer(text, return_tensors="pt")

    #predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    #get predicted class from logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return {{'predicted_class': predicted_class}}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
        """