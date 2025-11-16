from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from service.classification_service import get_prediction
import io
from PIL import Image
import numpy as np

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(), response_model=None):
    contents = await file.read()
    
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = get_prediction(img_array)
    return {"message": pred}

