from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from datetime import timedelta

MODEL_SAVE_PATH = 'weather_prediction_lstm_model.keras'
PREPROCESSOR_SAVE_PATH = 'weather_preprocessors.pkl'
FEATURES_LIST_SAVE_PATH = 'weather_feature_list.pkl'
DATA_PATH = 'https://drive.usercontent.google.com/download?id=1kzzLkqeBPVxUr-9Ec0GPsPR7UqtRdeMk&confirm=t'
TIME_STEPS = 7
TARGET_VARIABLES = ['precipprob', 'windspeed', 'temp', 'humidity']

model = None
feature_scaler = None
target_scaler = None
label_encoder = None
EXPECTED_FEATURES_ORDER = []
df_global = None


def engineer_features(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday

    if 'kecamatan' in df.columns:
        df['pressure_diff'] = df.groupby('kecamatan')['pressure'].diff().fillna(method='bfill').fillna(0)
    else:
        df['pressure_diff'] = df['pressure'].diff().fillna(method='bfill').fillna(0)

    df['dew_point_spread'] = df['temp'] - df['dew']

    LAG_PERIOD = 1
    humidity_lag_col = f'humidity_lag{LAG_PERIOD}'
    if 'kecamatan' in df.columns:
        df[humidity_lag_col] = df.groupby('kecamatan')['humidity'].shift(LAG_PERIOD)
    else:
        df[humidity_lag_col] = df['humidity'].shift(LAG_PERIOD)
    
    df[humidity_lag_col] = df[humidity_lag_col].fillna(method='bfill').fillna(method='ffill').fillna(df['humidity'].mean() if not df.empty else 0)

    return df

def load_and_preprocess_global_data():
    global df_global, label_encoder

    try:
        raw_df = pd.read_csv(DATA_PATH)
        print("Columns found in the DataFrame from URL:", raw_df.columns.tolist())
    except FileNotFoundError:
        print(f"Error: Data file {DATA_PATH} not found.")
        return False
    except Exception as e:
        print(f"Error reading data from {DATA_PATH}: {e}")
        return False
    
    df_eng = engineer_features(raw_df)
    
    df_global = df_eng
    print("Global DataFrame loaded and preprocessed.")

    
    if EXPECTED_FEATURES_ORDER and df_global is not None:
        cols_to_check_for_na = [col for col in EXPECTED_FEATURES_ORDER if col in df_global.columns]
        df_global.dropna(subset=cols_to_check_for_na, inplace=True)
        df_global.reset_index(drop=True, inplace=True)
        print(f"Global DataFrame shape after NA drop based on expected features: {df_global.shape}")


    return True


# FastAPI Application Setup
app = FastAPI(
    title="Weather Forecast API",
    description="API to predict weather conditions ('precipprob', 'windspeed', 'temp', 'humidity').",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    global model, feature_scaler, target_scaler, label_encoder, EXPECTED_FEATURES_ORDER
    try:
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print("Model loaded successfully.")
        
        with open(PREPROCESSOR_SAVE_PATH, 'rb') as f:
            preprocessors = pickle.load(f)
        feature_scaler = preprocessors['feature_scaler']
        target_scaler = preprocessors['target_scaler']
        label_encoder = preprocessors['label_encoder']
        print("Preprocessors loaded successfully.")

        with open(FEATURES_LIST_SAVE_PATH, 'rb') as f:
            EXPECTED_FEATURES_ORDER = pickle.load(f)
        print(f"Expected features order loaded: {EXPECTED_FEATURES_ORDER}")

        if not load_and_preprocess_global_data():
             raise RuntimeError("Failed to load or preprocess global data.")
        if df_global is None or df_global.empty:
            raise RuntimeError("Global dataframe is empty after loading and preprocessing.")


    except FileNotFoundError as e:
        print(f"Error loading model or preprocessors: {e}")
        model = None 
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        model = None 


class ForecastInput(BaseModel):
    kecamatan_name: str
    start_date: str
    forecast_days: int = 7 # Number of days to be forecasted

class ForecastOutput(BaseModel):
    datetime: str
    kecamatan: str
    precipprob: float
    windspeed: float
    temp: float
    humidity: float

# API Endpoints
@app.get("/")
def home():
    return {"message": "Weather Forecast API. Use the /predict endpoint to get forecasts."}

@app.post("/predict", response_model=List[ForecastOutput])
async def predict_weather(data: ForecastInput):
    try:
        # if not all([model, feature_scaler, target_scaler, label_encoder, EXPECTED_FEATURES_ORDER, df_global is not None]):
            # raise HTTPException(status_code=503, detail="Model, preprocessors, or global data not loaded. API is not ready.")

        try:
            forecast_start_dt = pd.to_datetime(data.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Please use YYYY-MM-DD.")

        kecamatan_df = df_global[df_global['kecamatan'] == data.kecamatan_name].sort_values('datetime').reset_index(drop=True)

        if kecamatan_df.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for kecamatan '{data.kecamatan_name}'.")

        history_end_dt = forecast_start_dt - timedelta(days=1)

        historical_sequence_df = kecamatan_df[kecamatan_df['datetime'] <= history_end_dt].tail(TIME_STEPS)

        if len(historical_sequence_df) < TIME_STEPS:
            raise HTTPException(status_code=400, 
                                detail=f"Not enough historical data for kecamatan '{data.kecamatan_name}' before {data.start_date}. "
                                       f"Need {TIME_STEPS} days, found {len(historical_sequence_df)}.")

        current_sequence_unscaled_df = historical_sequence_df.copy()

        try:
            kecamatan_encoded_val = label_encoder.transform([data.kecamatan_name])[0]
        except ValueError:
            raise HTTPException(status_code=400, 
                                detail=f"Kecamatan '{data.kecamatan_name}' not seen during training. Cannot encode.")
        X_cat_pred = np.array([[kecamatan_encoded_val]])

        all_forecast_outputs = []

        for day_idx in range(data.forecast_days):
            X_num_unscaled_for_pred = current_sequence_unscaled_df[EXPECTED_FEATURES_ORDER].values

            if X_num_unscaled_for_pred.shape != (TIME_STEPS, len(EXPECTED_FEATURES_ORDER)):
                 raise ValueError(f"Numerical input shape mismatch before scaling. Expected ({TIME_STEPS}, {len(EXPECTED_FEATURES_ORDER)}), "
                                     f"got {X_num_unscaled_for_pred.shape}. Check EXPECTED_FEATURES_ORDER and data preparation.")


            X_num_scaled_for_pred = feature_scaler.transform(X_num_unscaled_for_pred)
            X_seq_reshaped = X_num_scaled_for_pred.reshape((1, TIME_STEPS, len(EXPECTED_FEATURES_ORDER)))

            pred_scaled = model.predict([X_seq_reshaped, X_cat_pred], verbose=0)[0]

            pred_unscaled_targets_array = target_scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
            pred_unscaled_targets_dict = dict(zip(TARGET_VARIABLES, pred_unscaled_targets_array))

            actual_forecast_dt = forecast_start_dt + timedelta(days=day_idx)
            forecast_dt_str = actual_forecast_dt.strftime('%Y-%m-%d')

            all_forecast_outputs.append(
                ForecastOutput(
                    datetime=forecast_dt_str,
                    kecamatan=data.kecamatan_name,
                    precipprob=round(float(pred_unscaled_targets_dict['precipprob']), 4),
                    windspeed=round(float(pred_unscaled_targets_dict['windspeed']), 2),
                    temp=round(float(pred_unscaled_targets_dict['temp']), 2),
                    humidity=round(float(pred_unscaled_targets_dict['humidity']), 2)
                )
            )

            if day_idx == data.forecast_days - 1:
                break

            last_known_unscaled_row = current_sequence_unscaled_df.iloc[-1]
            next_input_datetime = actual_forecast_dt

            new_row_unscaled_dict = {}

            new_row_unscaled_dict['year'] = next_input_datetime.year
            new_row_unscaled_dict['month'] = next_input_datetime.month
            new_row_unscaled_dict['day'] = next_input_datetime.day
            new_row_unscaled_dict['weekday'] = next_input_datetime.weekday()

            if 'humidity_lag1' in EXPECTED_FEATURES_ORDER:
                new_row_unscaled_dict['humidity_lag1'] = pred_unscaled_targets_dict['humidity']


            carried_forward_dew = last_known_unscaled_row.get('dew', 0.0)
            if 'dew' in EXPECTED_FEATURES_ORDER:
                new_row_unscaled_dict['dew'] = carried_forward_dew

            if 'dew_point_spread' in EXPECTED_FEATURES_ORDER:
                new_row_unscaled_dict['dew_point_spread'] = pred_unscaled_targets_dict['temp'] - carried_forward_dew

            carried_forward_pressure = last_known_unscaled_row.get('pressure', 0.0)
            if 'pressure' in EXPECTED_FEATURES_ORDER:
                 new_row_unscaled_dict['pressure'] = carried_forward_pressure

            if 'pressure_diff' in EXPECTED_FEATURES_ORDER:
                new_row_unscaled_dict['pressure_diff'] = new_row_unscaled_dict.get('pressure', 0.0) - last_known_unscaled_row.get('pressure',0.0)

            for feat_name in EXPECTED_FEATURES_ORDER:
                if feat_name not in new_row_unscaled_dict:
                    new_row_unscaled_dict[feat_name] = last_known_unscaled_row.get(feat_name, 0.0) 

            new_row_df = pd.DataFrame([new_row_unscaled_dict])
            new_row_df['datetime'] = next_input_datetime 

            new_row_df['kecamatan'] = data.kecamatan_name 

            aligned_new_row_df = new_row_df.reindex(columns=current_sequence_unscaled_df.columns)


            current_sequence_unscaled_df = pd.concat([current_sequence_unscaled_df.iloc[1:], aligned_new_row_df], ignore_index=True)

        return all_forecast_outputs

    except ValueError as ve:
        print(f"ValueError during prediction: {str(ve)}") # Log for server
        raise HTTPException(status_code=400, detail=f"Input data processing error: {str(ve)}")

    except Exception as e:
        print(f"Unhandled exception during prediction: {e}") # Log for server
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
