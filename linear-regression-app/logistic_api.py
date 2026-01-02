from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import os

app = FastAPI()

# ✅ Allow frontend to call backend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logistic_model.joblib")
model = joblib.load(MODEL_PATH)

# Define expected input
class Features(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Logistic Regression API running"}

@app.post("/predict")
def predict(data: Features):
    X = np.array(data.features).reshape(1, -1)
    pred = int(model.predict(X)[0])
    result = "Malignant" if pred == 1 else "Benign"
    return {"model": "Logistic Regression", "prediction": result}
