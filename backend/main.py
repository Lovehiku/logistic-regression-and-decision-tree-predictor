from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained models
log_model = joblib.load("logistic_model.joblib")
tree_model = joblib.load("decision_tree_model.joblib")

# Pydantic model for request body
class Features(BaseModel):
    features: list[float]

# Root route so / doesn't give 404
@app.get("/")
def root():
    return {"message": "ML API is running! Go to /docs to test predictions."}

# Prediction endpoint
@app.post("/predict")
def predict(data: Features):
    try:
        # Convert features to numpy array and reshape
        X = np.array(data.features).reshape(1, -1)

        # Make predictions
        logistic_result = int(log_model.predict(X)[0])
        tree_result = int(tree_model.predict(X)[0])

        return {
            "logistic_prediction": logistic_result,
            "decision_tree_prediction": tree_result
        }
    except Exception as e:
        return {"error": str(e)}
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://127.0.0.1:5500",  # frontend origin (Live Server)
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
