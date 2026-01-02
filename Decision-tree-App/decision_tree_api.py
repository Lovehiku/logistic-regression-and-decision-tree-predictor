from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import joblib
import numpy as np
import os

app = FastAPI()

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_model.joblib")
model = joblib.load(MODEL_PATH)

# Input data structure
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
    return {"model": "Decision Tree", "prediction": result}

# Run server (Windows local & Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use Render PORT or fallback 8000
    uvicorn.run("logistic_api:app", host="0.0.0.0", port=port, reload=True)
