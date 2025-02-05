from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

print("📌 Loading model...")
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "ML Model API is Running"}

@app.post("/predict")
def predict(data: list):
    """Make predictions using the trained model"""
    prediction = model.predict(np.array(data))
    return {"prediction": prediction.tolist()}
