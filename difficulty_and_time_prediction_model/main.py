from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "best_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Load Scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "saved_model", "scaler.pkl")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Expected order of features
FEATURE_NAMES = ['ketinggian', 'jarak', 'elevation gain', 'temp', 'humidity', 'precipprob', 'windspeed']

app = FastAPI()

class InputData(BaseModel):
    ketinggian: float
    jarak: float
    elevation_gain: float
    temp: float
    humidity: float
    precipprob: float
    windspeed: float

@app.get("/")
def home():
    return {"message": "Difficulty & Time Estimation API"}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_dict = {
            "ketinggian": data.ketinggian,
            "jarak": data.jarak,
            "elevation gain": data.elevation_gain,
            "temp": data.temp,
            "humidity": data.humidity,
            "precipprob": data.precipprob,
            "windspeed": data.windspeed
        }

        input_df = np.array([[input_dict[feat] for feat in FEATURE_NAMES]])
        scaled_input = scaler.transform(input_df)
        pred_difficulty, pred_time = model.predict(scaled_input)[0]

        return {
            "predicted_difficulty_score": round(float(pred_difficulty), 2),
            "estimated_time_hours": round(float(pred_time), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
