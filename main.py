from fastapi import FastAPI
from pydantic import BaseModel
from model.model_fcn import predict_fcn
import numpy as np


class Features(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float 
    flipper_length_mm: float
    body_mass_g: float #| None = None


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}



@app.post("/predict/")
async def predict_species(features: Features):
    species=[]
    species=predict_fcn([features.culmen_length_mm,features.culmen_depth_mm,features.flipper_length_mm,features.body_mass_g]) # por qué no puedo enviar solo features? ¿cómo lo envío?
    return {"Your result was":species}