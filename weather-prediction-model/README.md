# Weather Forecasting Model and API Documentation

This document provides an overview of the weather forecasting model, the FastAPI-based API built to serve its predictions, and instructions on how to use the API. This project is intended for HiPlan.

---

## Table of Contents
1.  [Model Explanation](#model-explanation)
    1.  [Data and Features](#1-data-and-features)
    2.  [Preprocessing](#2-preprocessing)
    3.  [Model Architecture](#3-model-architecture)
    4.  [Training](#4-training)
2.  [API Explanation](#api-explanation)
    1.  [Initialization and Data Loading](#1-initialization-and-data-loading)
    2.  [Core Functionality](#2-core-functionality)
    3.  [API Endpoints](#3-api-endpoints)
3.  [How to Use the API](#how-to-use-the-api)
    1.  [Prerequisites](#prerequisites)
    2.  [Home Endpoint](#1-home-endpoint)
    3.  [Predict Endpoint](#2-predict-endpoint)

---

## Model Explanation

The model predicts future weather conditions, specifically **precipitation probability (`precipprob`)**, **wind speed (`windspeed`)**, **temperature (`temp`)**, and **humidity (`humidity`)**.

### 1. Data and Features

* **Dataset**: The model is trained on a time-series weather dataset. The API loads initial historical data from a specified URL (see API Startup).
* **Feature Engineering (`engineer_features` function)**:
    * **Datetime Features**: `year`, `month`, `day`, `weekday` are extracted from the `datetime` column.
    * **Differential Features**: `pressure_diff` (change in pressure, calculated per `kecamatan`) and `dew_point_spread` (difference between temperature and dew point: `temp - dew`) are calculated.
    * **Lagged Features**: `humidity_lag1` (humidity from the previous day, calculated per `kecamatan`) is created. Missing initial lag values are backfilled, then forward-filled, or filled with the mean humidity.
    * **Categorical Feature**: `kecamatan` (sub-district/area) is a key categorical input.
* **Target Variables**: `precipprob`, `windspeed`, `temp`, `humidity`.

### 2. Preprocessing

* **Cleaning**: During API startup, after feature engineering, rows with NA values in critical feature columns (defined by `EXPECTED_FEATURES_ORDER`) are dropped from the global historical dataset. The original training data also underwent missing value and duplicate removal.
* **Encoding**: The `kecamatan` categorical feature is converted into a numerical representation using a `LabelEncoder`.
* **Scaling**: Numerical features are scaled using `RobustScaler` (loaded as `feature_scaler`) to handle outliers effectively. Target variables are also scaled before training and their predictions are inverse-transformed (using `target_scaler`) after inference.
* **Sequencing**: The time-series data is transformed into sequences. The model uses a sequence of the past **7 days** (`TIME_STEPS = 7`) of numerical features to predict the weather for the next day.

### 3. Model Architecture

The model is a **Neural Network** built with TensorFlow/Keras, specifically designed for sequence data:

* **Inputs**:
    1.  `numerical_input`: A sequence of scaled numerical features for the past `TIME_STEPS` days. (Shape: `(TIME_STEPS, n_features)`)
    2.  `kecamatan_input`: The label-encoded `kecamatan` for which the prediction is being made. (Shape: `(1,)`)
* **Layers**:
    1.  **Embedding Layer**: The `kecamatan_input` is passed through an `Embedding` layer (`output_dim=8`) to create a dense vector representation. This is then `Flatten`ed and `RepeatVector`ed to match the `TIME_STEPS` dimension.
    2.  **Concatenation**: The processed numerical input and the repeated embedded categorical input are concatenated.
    3.  **Bidirectional LSTMs**:
        * First `Bidirectional LSTM` layer with 64 units (`return_sequences=True`) and L2 regularization.
        * `Dropout` (0.3).
        * Second `Bidirectional LSTM` layer with 32 units and L2 regularization.
        * `Dropout` (0.3).
    4.  **Dense Layers**:
        * A `Dense` layer with 32 units and 'relu' activation, with L2 regularization.
        * An output `Dense` layer with a number of units equal to the number of target variables (4), producing the final scaled predictions.
* **Compilation**:
    * Optimizer: `adam`
    * Loss Function: `mse` (Mean Squared Error)
    * Metrics: `MeanAbsoluteError`

### 4. Training

* The data is split into training and testing sets based on the year (e.g., data up to 2023 for training, after 2023 for testing).
* **Early Stopping**: Used during training to prevent overfitting by monitoring `val_loss` with a `patience` of 10.

---

## API Explanation

The API is built using FastAPI to serve a trained LSTM weather forecasting model. It's titled "Weather Forecast API" and provides daily predictions for precipitation probability, wind speed, temperature, and humidity for various sub-districts (kecamatan).

### 1. Initialization and Data Loading

When the application starts, several key assets are loaded into memory:

- Model Loading: A pre-trained Keras LSTM model (`weather_prediction_lstm_model.keras`) is loaded.
- Preprocessor Loading: A pickle file (weather_preprocessors.pkl) containing saved preprocessors is loaded. This includes:
  - `feature_scaler`: A scaler for the input features.
  - `target_scaler`: A scaler for the prediction targets.
  - `label_encoder`: An encoder for the categorical kecamatan feature.
- Feature List: The exact list of features (`weather_feature_list.pkl`) the model was trained on is loaded.
- Global Data Loading & Preparation:
  - Historical weather data is fetched from a Google Drive URL and loaded into a pandas DataFrame.
  - The engineer_features function is immediately applied to this DataFrame to create time-based features (`year`, `month`, `day`), interaction terms (`dew_point_spread`), and lagged variables (`humidity_lag1`).
  - This fully processed DataFrame (`df_featured`) serves as the in-memory database of historical data for all predictions.

If any of these files are not found, the application will raise a `RuntimeError` on startup.

### 2. Core Functionality

`engineer_features(df)`

A utility function that processes a raw weather DataFrame. It adds derived features essential for the model's performance, such as date components, pressure differences, dew point spread, and lagged humidity. This ensures that input data for prediction matches the training data format.

`forecast_from_date(...)`

This is the core prediction engine. It performs an iterative (auto-regressive) forecast:

1. It retrieves the last `MODEL_TIME_STEPS` (7) days of historical data for a given `kecamatan` before the specified `start_date`.
1. It then enters a loop for the number of `forecast_days`.
1. In each iteration, it uses the current 7-day sequence to predict the next day's weather.
1. Crucially, it then creates a new, synthetic data row for the day it just predicted. This new row includes the predicted values (`temp`, `humidity`, etc.) and uses them to calculate the necessary engineered features (like `dew_point_spread` and `humidity_lag1`) for the next prediction step.
1. This new synthetic row is added to the end of the sequence, and the oldest day is removed. The updated 7-day sequence is then used to predict the subsequent day.
1. This process repeats until the entire forecast period is covered.

### 3. API Endpoints

The API exposes two endpoints.

- `GET /`

  A simple root endpoint to confirm that the API is running.

  ```JSON
  {
    "message": "Weather Forecast API. Use the /forecast endpoint."
  }
  ```

- `POST /forecast`

  This is the main endpoint for generating weather forecasts.

  1. Request Body (`ForecastRequest`):
        
      - `kecamatan_name` (string, required): The name of the sub-district (e.g., "berastagi").
      - `start_date` (string, required, format "YYYY-MM-DD"): The first day for the forecast.
        
      - `forecast_days` (integer, required): The number of days to forecast (must be between 1 and 30).

  1. Processing Logic:
        
      - Validation: The endpoint first validates the request data.
      - Gap Handling: It intelligently handles requests for future dates. If a user requests a forecast starting on a date after the latest historical data available, the API will:
        - Start predicting from the day immediately following its last known data point.
        - Continue predicting day-by-day until it reaches the end of the user's requested forecast period.
        - Filter the results to only return the specific days the user originally requested.
      - Forecasting: It calls the `forecast_from_date` function with the appropriate parameters to generate the prediction sequence.
      - Response: The final forecast is returned as a list of JSON objects.
  1. Error Handling: The endpoint will return specific HTTP errors for common issues:
        - `400 Bad Request`: If the kecamatan name is not found, or if there is insufficient historical data to begin a forecast (i.e., less than 7 days of data before the start date).
        - `500 Internal Server Error`: For any other unexpected errors during the prediction process.


---

## How to Use the API

### Prerequisites

  Ensure the FastAPI server is running. You can start it with a command like:

  ```bash
    uvicorn main:app --reload
  ```

  The API will be available at http://127.0.0.1:8000.

#### 1. Check API Status (/)

  URL: /

  Method: GET

  Description: Checks if the API is operational.

  Example cURL:

  ```bash
    curl -X 'GET' 'http://127.0.0.1:8000/'
  ```

#### 2. Get Weather Forecast (/forecast)

  URL: /forecast

  Method: POST

  Description: Submits data to get a weather forecast for a specified kecamatan and date range.

  Example Request Body:


  ```json
  {
    "kecamatan_name": "berastagi",
    "start_date": "2025-08-08",
    "forecast_days": 5
  }
  ```

Example Request (using cURL):

```json
curl -X 'POST' \
  'http://127.0.0.1:8000/forecast' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "kecamatan_name": "berastagi",
    "start_date": "2025-08-08",
    "forecast_days": 5
  }'
```

Successful Response (`200 OK`):
A list of forecast objects, one for each day.

```json
[
  {
    "datetime": "2025-08-08",
    "kecamatan": "berastagi",
    "precipprob": 0.531,
    "windspeed": 9.85,
    "temp": 24.11,
    "humidity": 88.45
  },
  {
    "datetime": "2025-08-09",
    "kecamatan": "berastagi",
    "precipprob": 0.542,
    "windspeed": 9.92,
    "temp": 24.05,
    "humidity": 88.91
  },
  {
    "datetime": "2025-08-10",
    "kecamatan": "berastagi",
    "precipprob": 0.551,
    "windspeed": 10.01,
    "temp": 23.98,
    "humidity": 89.23
  },
  {
    "datetime": "2025-08-11",
    "kecamatan": "berastagi",
    "precipprob": 0.559,
    "windspeed": 10.15,
    "temp": 23.91,
    "humidity": 89.67
  },
  {
    "datetime": "2025-08-12",
    "kecamatan": "berastagi",
    "precipprob": 0.565,
    "windspeed": 10.22,
    "temp": 23.85,
    "humidity": 90.01
  }
]
```

*(Note: Actual values are examples and will vary based on the model's prediction.)*

Error Response (`400 Bad Request`):
If you provide a kecamatan that doesn't exist or doesn't have enough data.


```json
{
  "detail": "Kecamatan 'nonexistent_place' not found in the dataset."
}
```

```json
{
  "detail": "Not enough historical data for 'berastagi' before 2020-01-01. Need 7 days, but only found 0."
}
```