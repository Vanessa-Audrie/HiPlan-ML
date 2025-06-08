import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import calendar
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

# load model and preprocessor
try:
    SEASONAL_MODEL_PATH = 'weather_seasonal_model.keras'
    SEASONAL_PREPROCESSOR_PATH = 'weather_seasonal_preprocessor.pkl'
    
    MODEL = tf.keras.models.load_model(SEASONAL_MODEL_PATH)
    with open(SEASONAL_PREPROCESSOR_PATH, 'rb') as f:
        PREPROCESSOR = pickle.load(f)
    print("Model and preprocessor loaded successfully.")

except FileNotFoundError as e:
    print(f" ERROR: Could not load model or preprocessor file.")
    print(f"Make sure '{SEASONAL_MODEL_PATH}' and '{SEASONAL_PREPROCESSOR_PATH}' are in the same directory as main.py.")
    print(f"Details: {e}")
    exit()


# init fastapi app
app = FastAPI(
    title="Seasonal Weather Forecast API",
    description="An API to predict typical seasonal weather for a given location and date range.",
    version="1.0.0"
)

# date range prediction
def generate_seasonal_forecast(kecamatan_name: str, start_date_str: str, days_to_predict: int):
    """
    Internal function to generate weather forecasts. Uses the globally loaded model.
    """
    try:
        start_date = pd.to_datetime(start_date_str)
    except ValueError:
        return {"error": "Invalid date format. Please use 'YYYY-MM-DD'."}

    features_list = []
    date_range = [start_date + timedelta(days=i) for i in range(days_to_predict)]

    for date in date_range:
        features_list.append({
            'year': date.year,
            'day_sin': np.sin(2 * np.pi * date.dayofyear / 366),
            'day_cos': np.cos(2 * np.pi * date.dayofyear / 366),
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'kecamatan': kecamatan_name
        })
    
    input_df = pd.DataFrame(features_list)

    try:
        input_processed = PREPROCESSOR.transform(input_df)
    except Exception:
        return {"error": f"Could not process input. It's possible the location '{kecamatan_name}' was not in the training data."}

    all_predicted_values = MODEL.predict(input_processed)

    # format output
    final_results = []
    targets = ['precipprob', 'windspeed', 'temp', 'humidity']
    
    for i, date in enumerate(date_range):
        result = {'date': date.strftime('%Y-%m-%d'), 'kecamatan': kecamatan_name}
        predicted_values_for_day = all_predicted_values[i]
        for target, value in zip(targets, predicted_values_for_day):
            result[f"predicted_{target}"] = round(float(value), 2)
        final_results.append(result)
        
    return final_results





# monthly prediction
def generate_monthly_average(kecamatan_name: str, month: int, year: int):
    num_days = calendar.monthrange(year, month)[1]
    date_range = pd.to_datetime([f"{year}-{month:02d}-{day:02d}" for day in range(1, num_days + 1)])

    features_list = []
    for date in date_range:
        features_list.append({
            'year': date.year,
            'day_sin': np.sin(2 * np.pi * date.dayofyear / 366),
            'day_cos': np.cos(2 * np.pi * date.dayofyear / 366),
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'kecamatan': kecamatan_name
        })
    input_df = pd.DataFrame(features_list)

    try:
        input_processed = PREPROCESSOR.transform(input_df)
    except Exception:
        return {"error": f"Could not process input for '{kecamatan_name}'."}

    all_predicted_values = MODEL.predict(input_processed)
    average_values = np.mean(all_predicted_values, axis=0)
    
    targets = ['precipprob', 'windspeed', 'temp', 'humidity']

    result = {}
    for target, value in zip(targets, average_values):
        result[f"average_{target}"] = round(float(value), 2)

    return result

# threshold for determining weather
PRECIPPROB_THRESHOLD = 70.0
HUMIDITY_THRESHOLD = 92.0






# API ENDPOINTS

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


@app.get("/status")
def get_status():
    return {"status": "ok", "message": "API is running and model is loaded."}

@app.get("/forecast/range")
def get_forecast_range(kecamatan_name: str, start_date: str, days_to_predict: int):
    if days_to_predict < 1 or days_to_predict > 90:
        raise HTTPException(
            status_code=400, 
            detail="Days to predict must be between 1 and 90."
        )

    predictions = generate_seasonal_forecast(kecamatan_name, start_date, days_to_predict)

    if isinstance(predictions, dict) and "error" in predictions:
        raise HTTPException(status_code=400, detail=predictions["error"])

    return {
        "request_info": {
            "kecamatan_name": kecamatan_name,
            "start_date": start_date,
            "days_predicted": days_to_predict
        },
        "forecast": predictions
    }

@app.get("/forecast/monthly")
def get_monthly_forecast(kecamatan_name: str, month: int, year: int):
    """
    Provides the average seasonal weather forecast for a given month and year.
    """
    if not 1 <= month <= 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")
    if not 1970 <= year <= 2070:
        raise HTTPException(status_code=400, detail="Year must be between 1970 and 2070.")
        
    predictions = generate_monthly_average(kecamatan_name, month, year)
    if isinstance(predictions, dict) and "error" in predictions:
        raise HTTPException(status_code=400, detail=predictions["error"])
        
    return {
        "request_info": {"kecamatan_name": kecamatan_name, "month": calendar.month_name[month], "year": year},
        "forecast": predictions
    }


@app.get("/forecast/seasonality")
def get_seasonality_forecast(kecamatan_name: str, month: int, year: int):

    if not 1 <= month <= 12:
        raise HTTPException(status_code=400, detail="Month must be between 1 and 12.")
    if not 1970 <= year <= 2070:
        raise HTTPException(status_code=400, detail="Year must be between 1970 and 2070.")
        
    # call the monthly function
    monthly_avg = generate_monthly_average(kecamatan_name, month, year)
    if isinstance(monthly_avg, dict) and "error" in monthly_avg:
        raise HTTPException(status_code=400, detail=monthly_avg["error"])
        

    if (monthly_avg['average_precipprob'] > PRECIPPROB_THRESHOLD or monthly_avg['average_humidity'] > HUMIDITY_THRESHOLD):
        seasonality = "Hujan"
    else:
        seasonality = "Cerah"
        
    return {
        "request_info": {"kecamatan_name": kecamatan_name, "month": calendar.month_name[month], "year": year},
        "analysis": {
            "determined_seasonality": seasonality,
            "reasoning_metrics": monthly_avg
        }
    }