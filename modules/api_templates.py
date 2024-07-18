'''Module for registering API templates to use for deploy'''

class APITemplates:
    '''Class to save API templates'''

    def __init__(self) -> None:
        self.buildTemplate1()
    
    def getFilledTemplate1(self, model_path, tokenizer_path, port):
        return self.template_1.format(model_path, tokenizer_path, port)
    
    def buildTemplate1(self):
        self.template_1 = """
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
import torch


app = FastAPI()

tokenizer = None
model = None

async def loadTokenizer():
    '''Async. function to load tokenizer'''

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained('{}')

async def loadModel():
    '''Async. function to load model'''

    global model
    model = AutoModelForSequenceClassification.from_pretrained('{}')

async def runTokenization(text):
    '''Async. function to tokenize'''

    global tokenizer
    await loadTokenizer() #ensure tokenizer is loaded

    #tokenize
    inputs = tokenizer(text, return_tensors="pt")

    return inputs

async def runInference(inputs):
    '''Async. function to infer'''

    global model
    await loadModel() #ensure model is loaded

    #infer
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    return logits

#request data type
class PredictionRequest(BaseModel):
    text: str

@app.post('/predict')
async def predict(request_data: PredictionRequest):
    #get request text
    text = request_data.text
    
    #tokenize text
    inputs = await runTokenization(text)

    #predict
    logits = await runInference(inputs)
    
    #get predicted class from logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return {{'predicted_class': predicted_class}}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port={})
        """