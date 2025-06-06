import pickle
import uvicorn
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date, timedelta

# --- 1. App & Model Loading ---
app = FastAPI(
    title="Ideal Weather Forecast API",
    description="An API to forecast weather using a self-sufficient LSTM model.",
    version="4.0.0"
)

# Paths to the artifacts created by the training script
MODEL_PATH = 'weather_prediction_lstm_model.keras'
PREPROCESSOR_PATH = 'ideal_preprocessors.pkl'
DATA_PATH = 'https://drive.usercontent.google.com/uc?id=1WZrk8tYPzErxdSoLOttI0xlsteJunz5X'
MODEL_TIME_STEPS = 7

# Load all artifacts in a try-except block
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessors = pickle.load(f)
        
    scaler = preprocessors['scaler']
    label_encoder = preprocessors['label_encoder']
    features_list = preprocessors['features_list']
    
    # Load historical data needed to start the prediction sequence
    df_raw = pd.read_csv(DATA_PATH)

except FileNotFoundError as e:
    raise RuntimeError(f"Could not load a required file: {e}. Run train_model.py first.")

# --- 2. Feature Engineering (must be identical to training) ---
def engineer_features(df):
    df_processed = df.copy()
    df_processed['datetime'] = pd.to_datetime(df_processed['datetime'])
    df_processed['year'] = df_processed['datetime'].dt.year
    df_processed['month'] = df_processed['datetime'].dt.month
    df_processed['day'] = df_processed['datetime'].dt.day
    df_processed['weekday'] = df_processed['datetime'].dt.weekday
    df_processed = df_processed.sort_values(by=['kecamatan', 'datetime'])
    df_processed['pressure_diff'] = df_processed.groupby('kecamatan')['pressure'].diff().fillna(0)
    df_processed['dew_point_spread'] = df_processed['temp'] - df_processed['dew']
    LAG_PERIOD = 1
    df_processed[f'humidity_lag{LAG_PERIOD}'] = df_processed.groupby('kecamatan')['humidity'].shift(LAG_PERIOD)
    df_processed = df_processed.dropna().reset_index(drop=True)
    return df_processed

# Prepare the historical data once on startup
df_featured = engineer_features(df_raw)


# --- 3. Pydantic Models for Request & Response ---
class ForecastRequest(BaseModel):
    kecamatan_name: str = Field(..., example="berastagi", description="The name of the sub-district.")
    start_date: date = Field(..., example="2025-08-08", description="Start date for the forecast (YYYY-MM-DD).")
    forecast_days: int = Field(7, gt=0, le=30, example=7, description="Number of days to forecast (1-30).")

class ForecastResponse(BaseModel):
    datetime: date
    kecamatan: str
    precipprob: float
    windspeed: float
    temp: float
    humidity: float

# --- 4. Prediction Function ---
def forecast_from_date_ideal(kecamatan_name: str, start_date: date, forecast_days: int) -> pd.DataFrame:
    start_date_dt = pd.to_datetime(start_date)
    
    kecamatan_data = df_featured[df_featured['kecamatan'] == kecamatan_name]
    historical_data = kecamatan_data[kecamatan_data['datetime'] < start_date_dt]
    
    if len(historical_data) < MODEL_TIME_STEPS:
        raise ValueError(f"Not enough historical data for {kecamatan_name} before {start_date}.")

    current_sequence_df = historical_data.tail(MODEL_TIME_STEPS).copy()
    kecamatan_encoded_val = label_encoder.transform([kecamatan_name])[0]
    
    final_forecasts = []
    
    for i in range(forecast_days):
        sequence_scaled = scaler.transform(current_sequence_df[features_list].values)
        
        X_seq_num = sequence_scaled.reshape(1, MODEL_TIME_STEPS, len(features_list))
        X_seq_cat = np.array([[kecamatan_encoded_val]])
        
        prediction_scaled = model.predict([X_seq_num, X_seq_cat], verbose=0)
        prediction_inversed = scaler.inverse_transform(prediction_scaled)
        
        new_row = pd.DataFrame(prediction_inversed, columns=features_list)
        new_row['datetime'] = start_date_dt + pd.Timedelta(days=i)
        new_row['kecamatan'] = kecamatan_name
        
        current_sequence_df = pd.concat([current_sequence_df.iloc[1:], new_row], ignore_index=True)
        final_forecasts.append(new_row)

    forecast_df = pd.concat(final_forecasts, ignore_index=True)
    return forecast_df

# --- 5. API Endpoints ---
@app.get("/")
def home():
    return {"message": "Ideal Weather Forecast API. Use the /forecast endpoint."}

@app.post("/forecast", response_model=list[ForecastResponse])
async def create_forecast(request: ForecastRequest):
    try:
        forecast_result_df = forecast_from_date_ideal(
            kecamatan_name=request.kecamatan_name,
            start_date=request.start_date,
            forecast_days=request.forecast_days
        )
        # Select only the columns needed for the response model before returning
        response_cols = ['datetime', 'kecamatan', 'precipprob', 'windspeed', 'temp', 'humidity']
        return forecast_result_df[response_cols].to_dict(orient="records")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the detailed error on the server for debugging
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# To run the app, use the command: uvicorn main:app --reload