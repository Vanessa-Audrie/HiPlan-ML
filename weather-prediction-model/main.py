import pickle
import uvicorn
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date, timedelta
import asyncio

# -------------------------------------------------------------------
#  1. App & Asset Paths
# -------------------------------------------------------------------

app = FastAPI(
    title="Weather Forecast API",
    description="An API to forecast weather using a trained LSTM model with lazy loading.",
    version="3.1.0"
)

# --- Asset Paths ---
MODEL_PATH = 'weather_prediction_lstm_model.keras'
PREPROCESSOR_PATH = 'weather_preprocessors.pkl'
FEATURES_PATH = 'weather_feature_list.pkl'
DATA_PATH = 'https://drive.usercontent.google.com/uc?id=1WZrk8tYPzErxdSoLOttI0xlsteJunz5X'

# --- Model Constants ---
LAG_PERIOD = 1
MODEL_TIME_STEPS = 7

# --- Lazy Loading Cache ---
# This dictionary will hold our model and data once they are loaded.
_cache = {}
# A lock to prevent race conditions during the first load in a concurrent environment
_load_lock = asyncio.Lock()

# -------------------------------------------------------------------
#  2. Lazy Loading and Feature Engineering
# -------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the same feature engineering as the training notebook.
    """
    df_processed = df.copy()
    df_processed['datetime'] = pd.to_datetime(df_processed['datetime'])
    
    df_processed['year'] = df_processed['datetime'].dt.year
    df_processed['month'] = df_processed['datetime'].dt.month
    df_processed['day'] = df_processed['datetime'].dt.day
    df_processed['weekday'] = df_processed['datetime'].dt.weekday

    df_processed = df_processed.sort_values(by=['kecamatan', 'datetime'])
    df_processed['pressure_diff'] = df_processed.groupby('kecamatan')['pressure'].diff().fillna(0)
    df_processed['dew_point_spread'] = df_processed['temp'] - df_processed['dew']
    
    df_processed[f'humidity_lag{LAG_PERIOD}'] = df_processed.groupby('kecamatan')['humidity'].shift(LAG_PERIOD)
    
    df_processed = df_processed.dropna().reset_index(drop=True)
    
    return df_processed

async def load_assets():
    """
    Loads all assets lazily. If assets are already in the cache, this function returns immediately.
    This function is now async to handle the lock properly.
    """
    async with _load_lock:
        # Check if assets are already loaded to avoid redundant I/O
        if "model" not in _cache:
            print("Loading assets for the first time...")
            try:
                # Load the model
                _cache['model'] = tf.keras.models.load_model(MODEL_PATH)

                # Load preprocessors
                with open(PREPROCESSOR_PATH, 'rb') as f:
                    preprocessors = pickle.load(f)
                _cache['feature_scaler'] = preprocessors['feature_scaler']
                _cache['target_scaler'] = preprocessors['target_scaler']
                _cache['label_encoder'] = preprocessors['label_encoder']

                # Load feature list
                with open(FEATURES_PATH, 'rb') as f:
                    _cache['features'] = pickle.load(f)

                # Load and process data
                df_raw = pd.read_csv(DATA_PATH)
                _cache['df_featured'] = engineer_features(df_raw)
                
                print("Assets loaded successfully.")

            except FileNotFoundError as e:
                # If a file is missing, we cannot proceed.
                raise RuntimeError(f"Could not load a required file: {e}. Ensure all model artifacts are present.")
            except Exception as e:
                # Clear cache on any other loading error
                _cache.clear()
                raise RuntimeError(f"An unexpected error occurred while loading assets: {e}")

# -------------------------------------------------------------------
#  3. Pydantic Models
# -------------------------------------------------------------------

class ForecastRequest(BaseModel):
    kecamatan_name: str = Field(..., example="berastagi", description="The name of the sub-district (kecamatan).")
    start_date: date = Field(..., example="2025-08-08", description="The start date for the forecast (YYYY-MM-DD).")
    forecast_days: int = Field(7, gt=0, le=30, example=7, description="Number of days to forecast (1-30).")

class ForecastResponse(BaseModel):
    datetime: date
    kecamatan: str
    precipprob: float
    windspeed: float
    temp: float
    humidity: float

# -------------------------------------------------------------------
#  4. Forecasting Logic
# -------------------------------------------------------------------

def forecast_from_date(
    kecamatan_name: str,
    start_date_str: str,
    forecast_days: int
) -> pd.DataFrame:
    
    targets = ['precipprob', 'windspeed', 'temp', 'humidity']
    start_date_dt = pd.to_datetime(start_date_str)
    
    df_featured = _cache['df_featured']
    kecamatan_data = df_featured[df_featured['kecamatan'] == kecamatan_name].sort_values('datetime').reset_index(drop=True)

    if kecamatan_data.empty:
        raise ValueError(f"Kecamatan '{kecamatan_name}' not found in the dataset.")

    historical_data = kecamatan_data[kecamatan_data['datetime'] < start_date_dt]

    if len(historical_data) < MODEL_TIME_STEPS:
        raise ValueError(
            f"Not enough historical data for '{kecamatan_name}' before {start_date_dt.date()}. "
            f"Need {MODEL_TIME_STEPS} days, but only found {len(historical_data)}."
        )

    current_sequence_df = historical_data.tail(MODEL_TIME_STEPS).copy()
    
    try:
        kecamatan_encoded_val = _cache['label_encoder'].transform([kecamatan_name])[0]
    except ValueError:
       raise ValueError(f"Kecamatan '{kecamatan_name}' not in the model's vocabulary.")

    forecast_outputs = []
    
    original_cols = current_sequence_df.columns.tolist()
    features = _cache['features']
    
    for i in range(forecast_days):
        future_date = start_date_dt + timedelta(days=i)
        
        sequence_scaled = _cache['feature_scaler'].transform(current_sequence_df[features].values)
        X_seq_num = sequence_scaled.reshape(1, MODEL_TIME_STEPS, len(features))
        X_seq_cat = np.array([[kecamatan_encoded_val]])
        
        prediction_scaled = _cache['model'].predict([X_seq_num, X_seq_cat], verbose=0)
        prediction_inversed = _cache['target_scaler'].inverse_transform(prediction_scaled)
        
        forecast_outputs.append(prediction_inversed[0])

        last_row = current_sequence_df.iloc[-1]
        
        new_row_data = {col: val for col, val in zip(targets, prediction_inversed[0])}
        
        new_row_data['datetime'] = future_date
        new_row_data['kecamatan'] = kecamatan_name
        new_row_data['year'] = future_date.year
        new_row_data['month'] = future_date.month
        new_row_data['day'] = future_date.day
        new_row_data['weekday'] = future_date.weekday()
        new_row_data['dew'] = last_row['dew']
        new_row_data['pressure_diff'] = new_row_data.get('pressure', last_row['pressure']) - last_row['pressure']
        new_row_data['dew_point_spread'] = new_row_data['temp'] - new_row_data['dew']
        new_row_data[f'humidity_lag{LAG_PERIOD}'] = last_row['humidity']
        
        for col in original_cols:
            if col not in new_row_data and col not in features:
                new_row_data[col] = last_row[col]

        new_row_df = pd.DataFrame([new_row_data])
        new_row_df = new_row_df.reindex(columns=original_cols, fill_value=0)
        current_sequence_df = pd.concat([current_sequence_df.iloc[1:], new_row_df], ignore_index=True)

    forecast_df = pd.DataFrame(np.array(forecast_outputs), columns=targets)
    forecast_dates = pd.date_range(start=start_date_str, periods=forecast_days)
    forecast_df['datetime'] = forecast_dates
    forecast_df['kecamatan'] = kecamatan_name

    return forecast_df[['datetime', 'kecamatan'] + targets]

# -------------------------------------------------------------------
#  5. API Endpoints
# -------------------------------------------------------------------

@app.get("/")
def home():
    return {"message": "Weather Forecast API is running. Use the /forecast endpoint to get predictions."}

@app.post("/forecast", response_model=list[ForecastResponse])
async def create_forecast(request: ForecastRequest):
    """
    Creates a weather forecast. Assets will be loaded on the first request.
    """
    # This will ensure all assets are loaded before proceeding.
    # It will only perform the load operation on the very first API call.
    await load_assets()
    
    try:
        df_featured = _cache['df_featured']
        last_data_date = df_featured['datetime'].max().date()
        request_start_date = request.start_date

        if request_start_date > last_data_date:
            model_start_date = last_data_date + timedelta(days=1)
            total_days_to_predict = (request_start_date - last_data_date).days + request.forecast_days
            
            full_forecast_df = forecast_from_date(
                kecamatan_name=request.kecamatan_name,
                start_date_str=model_start_date.strftime('%Y-%m-%d'),
                forecast_days=total_days_to_predict
            )
            
            full_forecast_df['datetime'] = pd.to_datetime(full_forecast_df['datetime']).dt.date
            final_forecast_df = full_forecast_df[
                full_forecast_df['datetime'] >= request_start_date
            ].head(request.forecast_days)
            
            return final_forecast_df.to_dict(orient="records")
        
        else:
            start_date_str = request.start_date.strftime('%Y-%m-%d')
            forecast_result_df = forecast_from_date(
                kecamatan_name=request.kecamatan_name,
                start_date_str=start_date_str,
                forecast_days=request.forecast_days
            )
            forecast_result_df['datetime'] = pd.to_datetime(forecast_result_df['datetime']).dt.date
            return forecast_result_df.to_dict(orient="records")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # It's good practice to log the actual error for debugging
        print(f"An unexpected error occurred: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")