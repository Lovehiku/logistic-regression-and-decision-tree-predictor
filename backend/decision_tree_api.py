from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_model.joblib")
model = joblib.load(MODEL_PATH)

class Features(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Decision Tree API running"}

@app.post("/predict")
def predict(data: Features):
    X = np.array(data.features).reshape(1, -1)
    pred = int(model.predict(X)[0])

    result = "Malignant" if pred == 1 else "Benign"

    return {
        "model": "Decision Tree",
        "prediction": result
    }
