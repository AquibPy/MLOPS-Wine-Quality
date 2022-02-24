import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

app = FastAPI()
classifier=joblib.load('saved_models/model.joblib')

@app.get('/')
def get_root():
    return {'message': 'Welcome to the Wine quality API'}

@app.post('/predict')
def predict_quality(request:Wine):
    request = request.dict()
    fixed_acidity = request['fixed_acidity']
    volatile_acidity = request['volatile_acidity']
    citric_acid = request['citric_acid']
    residual_sugar = request['residual_sugar']
    chlorides = request['chlorides']
    free_sulfur_dioxide = request['free_sulfur_dioxide']
    total_sulfur_dioxide = request['total_sulfur_dioxide']
    density = request['density']
    pH = request['pH']
    sulphates = request['sulphates']
    alcohol = request['alcohol']

    prediction = classifier.predict([[
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol]])
    output = prediction
    return {'Data Recieved':request,'prediction': output}


if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000)