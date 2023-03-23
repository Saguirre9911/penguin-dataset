from fastapi import FastAPI
from pydantic import BaseModel


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
    return features