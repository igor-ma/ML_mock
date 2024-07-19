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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pydantic import BaseModel


app = FastAPI()

tokenizer = None
model = None

async def loadTokenizer():
    '''Async. function to load tokenizer'''

    global tokenizer
    tokenizer = T5Tokenizer.from_pretrained('{}')

async def loadModel():
    '''Async. function to load model'''

    global model
    model = T5ForConditionalGeneration.from_pretrained('{}')

async def runTokenization(text):
    '''Async. function to tokenize'''

    global tokenizer
    await loadTokenizer() #ensure tokenizer is loaded

    #tokenize
    input_ids = tokenizer(text, return_tensors='pt').input_ids

    return input_ids

async def runInference(input_ids):
    '''Async. function to infer'''

    global model, tokenizer
    await loadModel() #ensure model is loaded

    #infer
    output = model.generate(input_ids)

    #decode
    decoded_output = tokenizer.decode(output[0])
    
    return decoded_output

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
    decoded_output = await runInference(inputs)
    
    return {{'response': decoded_output}}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port={})
        """