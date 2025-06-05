from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import joblib
import os
import logging
import json
from saved_model.utils import hours_to_hh_mm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "best_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "saved_model", "minmax_scaler.pkl")
FEATURE_LIST_PATH = os.path.join(BASE_DIR, "saved_model", "feature_list.json")

# Load feature list from JSON
try:
    with open(FEATURE_LIST_PATH, "r") as f:
        FEATURE_NAMES = json.load(f)
    logger.info(f"Feature list loaded: {FEATURE_NAMES}")
except Exception as e:
    logger.error(f"Failed to load feature list: {e}")
    raise RuntimeError("Failed to load feature list.")

# Load scaler
try:
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load scaler: {e}")
    raise RuntimeError("Failed to load scaler.")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Failed to load model.")

# Request schema
class InputData(BaseModel):
    ketinggian: float
    jarak: float
    elevation_gain: float
    temp: float
    precipprob: float
    windspeed: float
    humidity: float

# Routes
@app.get("/")
def home():
    return {"message": "Difficulty & Time Estimation API"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Prepare data in the correct order based on loaded feature list
        input_features = [getattr(data, feature.replace(" ", "_")) for feature in FEATURE_NAMES]
        input_array = np.array([input_features])

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Predict
        prediction = model.predict(scaled_input)
        if prediction.shape[1] != 2:
            raise ValueError("Model must output two values: difficulty and estimated time.")

        pred_difficulty, pred_time = prediction[0]

        return {
            "estimated_time": hours_to_hh_mm(pred_time),
            "difficulty_score": round(float(pred_difficulty), 2)
        }

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
