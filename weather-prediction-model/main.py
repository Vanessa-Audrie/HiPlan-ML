from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from datetime import datetime, timedelta

MODEL_SAVE_PATH = 'weather_prediction_lstm_model.keras'
PREPROCESSOR_SAVE_PATH = 'weather_preprocessors.pkl'
FEATURES_LIST_SAVE_PATH = 'weather_feature_list.pkl'
TIME_STEPS = 7 # This must match the TIME_STEPS used during training

try:
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    
    with open(PREPROCESSOR_SAVE_PATH, 'rb') as f:
        preprocessors = pickle.load(f)
    feature_scaler = preprocessors['feature_scaler']
    target_scaler = preprocessors['target_scaler']
    label_encoder = preprocessors['label_encoder']

    with open(FEATURES_LIST_SAVE_PATH, 'rb') as f:
        EXPECTED_FEATURES_ORDER = pickle.load(f) 

except FileNotFoundError as e:
    print(f"Error loading model or preprocessors: {e}")
    print("Please ensure the model, preprocessors, and feature list files are in the correct directory.")

    model = None 
    feature_scaler = None
    target_scaler = None
    label_encoder = None
    EXPECTED_FEATURES_ORDER = []


app = FastAPI(
    title="Weather Forecast API 🌦️",
    description="API to predict weather conditions ('precipprob', 'windspeed', 'temp', 'humidity').",
    version="1.0.0"
)


class HistoricalDataPoint(BaseModel):
    features: Dict[str, float] 

class ForecastInput(BaseModel):
    kecamatan_name: str
    historical_sequence: conlist(HistoricalDataPoint, min_length=TIME_STEPS, max_length=TIME_STEPS)
    forecast_days: int = 1 

class ForecastOutput(BaseModel):
    datetime: str
    kecamatan: str
    precipprob: float
    windspeed: float
    temp: float
    humidity: float

# --- API Endpoints ---

@app.get("/")
def home():
    return {"message": "Weather Forecast API. Use the /predict endpoint to get forecasts."}

@app.post("/predict", response_model=List[ForecastOutput])
async def predict_weather(data: ForecastInput):
    if not model or not feature_scaler or not target_scaler or not label_encoder or not EXPECTED_FEATURES_ORDER:
        raise HTTPException(status_code=503, detail="Model or preprocessors not loaded. API is not ready.")

    try:
        # Prepare numerical features from the historical sequence
        numerical_input_list = []
        for i in range(TIME_STEPS):
            point_features = data.historical_sequence[i].features
            # Ensure features are in the correct order
            ordered_feature_values = [point_features.get(feat_name, 0.0) for feat_name in EXPECTED_FEATURES_ORDER]
            numerical_input_list.append(ordered_feature_values)
        
        X_num_unscaled = np.array(numerical_input_list)

        if X_num_unscaled.shape != (TIME_STEPS, len(EXPECTED_FEATURES_ORDER)):
            raise ValueError(f"Numerical input shape mismatch. Expected ({TIME_STEPS}, {len(EXPECTED_FEATURES_ORDER)}), "
                             f"got {X_num_unscaled.shape}")

        X_num_scaled = feature_scaler.transform(X_num_unscaled)
        X_num_reshaped = X_num_scaled.reshape((1, TIME_STEPS, len(EXPECTED_FEATURES_ORDER)))

        # Prepare categorical feature (kecamatan)
        try:
            kecamatan_encoded_val = label_encoder.transform([data.kecamatan_name])[0]
        except ValueError:
            raise HTTPException(status_code=400, 
                                detail=f"Kecamatan '{data.kecamatan_name}' not seen during training. Cannot encode.")
        X_cat_pred = np.array([[kecamatan_encoded_val]])

        # 3. Make predictions (iteratively for forecast_days)

        current_X_num_scaled_sequence = X_num_scaled # This is (TIME_STEPS, n_features)
        
        if data.forecast_days == 1:
            pred_scaled = model.predict([X_num_reshaped, X_cat_pred], verbose=0)[0]
            pred_unscaled_targets = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
            
            forecast_results = [
                ForecastOutput(
                    datetime="T+1", # Placeholder date
                    kecamatan=data.kecamatan_name,
                    precipprob=round(float(pred_unscaled_targets[0]), 4),
                    windspeed=round(float(pred_unscaled_targets[1]), 2),
                    temp=round(float(pred_unscaled_targets[2]), 2),
                    humidity=round(float(pred_unscaled_targets[3]), 2)
                )
            ]
            return forecast_results
        else:
            all_forecast_outputs = []

            
            temp_historical_data = []
            base_date_for_sequence = datetime.now() # Placeholder
            for idx, hist_point in enumerate(data.historical_sequence):
                row = hist_point.features.copy()
                
                row['datetime'] = base_date_for_sequence - timedelta(days=(TIME_STEPS - 1 - idx))
                temp_historical_data.append(row)
            
            current_sequence_unscaled_df = pd.DataFrame(temp_historical_data)
            
            current_sequence_unscaled_df = current_sequence_unscaled_df[list(current_sequence_unscaled_df.columns.drop('datetime')) + ['datetime']]


            for day_idx in range(data.forecast_days):
                
                X_num_unscaled_for_pred_iter = current_sequence_unscaled_df[EXPECTED_FEATURES_ORDER].values
                X_num_scaled_for_pred_iter = feature_scaler.transform(X_num_unscaled_for_pred_iter)
                X_seq_reshaped_iter = X_num_scaled_for_pred_iter.reshape((1, TIME_STEPS, len(EXPECTED_FEATURES_ORDER)))

                pred_scaled_iter = model.predict([X_seq_reshaped_iter, X_cat_pred], verbose=0)[0]
                pred_unscaled_targets_array_iter = target_scaler.inverse_transform(pred_scaled_iter.reshape(1, -1))[0]
                pred_unscaled_targets_dict_iter = dict(zip(['precipprob', 'windspeed', 'temp', 'humidity'], pred_unscaled_targets_array_iter))

                
                forecast_dt_str = f"T+{day_idx+1}"
                if 'datetime' in current_sequence_unscaled_df.columns:
                    actual_forecast_dt = current_sequence_unscaled_df['datetime'].iloc[-1] + timedelta(days=1)
                    forecast_dt_str = actual_forecast_dt.strftime('%Y-%m-%d')


                all_forecast_outputs.append(
                    ForecastOutput(
                        datetime=forecast_dt_str, # Placeholder
                        kecamatan=data.kecamatan_name,
                        precipprob=round(float(pred_unscaled_targets_dict_iter['precipprob']), 4),
                        windspeed=round(float(pred_unscaled_targets_dict_iter['windspeed']), 2),
                        temp=round(float(pred_unscaled_targets_dict_iter['temp']), 2),
                        humidity=round(float(pred_unscaled_targets_dict_iter['humidity']), 2)
                    )
                )

                if day_idx == data.forecast_days - 1:
                    break

                
                last_known_unscaled_row = current_sequence_unscaled_df.iloc[-1]
                next_forecast_dt_for_sequence = current_sequence_unscaled_df['datetime'].iloc[-1] + pd.Timedelta(days=1)
                
                new_row_unscaled_dict = {}
                new_row_unscaled_dict['year'] = next_forecast_dt_for_sequence.year
                new_row_unscaled_dict['month'] = next_forecast_dt_for_sequence.month
                new_row_unscaled_dict['day'] = next_forecast_dt_for_sequence.day
                new_row_unscaled_dict['weekday'] = next_forecast_dt_for_sequence.weekday()

                if 'humidity_lag1' in EXPECTED_FEATURES_ORDER:
                    new_row_unscaled_dict['humidity_lag1'] = pred_unscaled_targets_dict_iter['humidity']
                
                carried_forward_dew = last_known_unscaled_row.get('dew', 0.0)
                if 'dew' in EXPECTED_FEATURES_ORDER:
                     new_row_unscaled_dict['dew'] = carried_forward_dew
                if 'dew_point_spread' in EXPECTED_FEATURES_ORDER:
                    new_row_unscaled_dict['dew_point_spread'] = pred_unscaled_targets_dict_iter['temp'] - carried_forward_dew
                
                carried_forward_pressure = last_known_unscaled_row.get('pressure', 0.0)
                if 'pressure' in EXPECTED_FEATURES_ORDER:
                    new_row_unscaled_dict['pressure'] = carried_forward_pressure
                if 'pressure_diff' in EXPECTED_FEATURES_ORDER:
                    new_row_unscaled_dict['pressure_diff'] = carried_forward_pressure - last_known_unscaled_row.get('pressure', carried_forward_pressure)

                for feat_name in EXPECTED_FEATURES_ORDER:
                    if feat_name not in new_row_unscaled_dict:
                        new_row_unscaled_dict[feat_name] = last_known_unscaled_row.get(feat_name, 0.0)
                
                new_row_df = pd.DataFrame([new_row_unscaled_dict])
                new_row_df['datetime'] = next_forecast_dt_for_sequence # Add datetime for consistency
                # new_row_df['kecamatan'] = data.kecamatan_name # Not part of numerical features

                current_sequence_unscaled_df = pd.concat([current_sequence_unscaled_df.iloc[1:], new_row_df], ignore_index=True)
            
            return all_forecast_outputs


    except ValueError as ve: # Catch specific errors for better client feedback
        raise HTTPException(status_code=400, detail=f"Input data error: {str(ve)}")
    except Exception as e:
        # Log the exception e for server-side debugging
        print(f"Unhandled exception: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# To run the API (save this as main.py):
# uvicorn main:app --reload