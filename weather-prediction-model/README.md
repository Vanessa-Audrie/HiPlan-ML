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
    1.  [Startup](#1-startup)
    2.  [Core Functionality](#2-core-functionality)
    3.  [/predict Endpoint Logic](#3-predict-endpoint-logic)
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

The API is built using **FastAPI** to serve the trained weather forecasting model. It's titled "Weather Forecast API" and provides predictions for precipitation probability, wind speed, temperature, and humidity.

### 1. Startup

Triggered by `@app.on_event("startup")`:

* **Model Loading**: The pre-trained Keras model (`weather_prediction_lstm_model.keras`) is loaded.
* **Preprocessor Loading**: Saved preprocessors (`weather_preprocessors.pkl`), which include the `feature_scaler` (RobustScaler), `target_scaler` (RobustScaler), and `label_encoder` (LabelEncoder for `kecamatan`), are loaded using `pickle`.
* **Feature List Loading**: The specific list and order of numerical features (`EXPECTED_FEATURES_ORDER`) used during training (`weather_feature_list.pkl`) are loaded.
* **Global Data Loading & Preprocessing**:
    * Historical weather data is loaded from a Google Drive URL (`DATA_PATH`) into a global pandas DataFrame (`df_global`).
    * The `engineer_features` function (described in [Model Explanation](#1-data-and-features)) is applied to this DataFrame.
    * Rows with NA values in any of the `EXPECTED_FEATURES_ORDER` columns are dropped.
    * This `df_global` serves as the source of historical data for making new predictions.
* If any of these steps fail, the API might not function correctly, potentially leading to errors when endpoints are called.

### 2. Core Functionality

* **Feature Engineering (`engineer_features`)**: A utility function that takes a DataFrame and adds derived features like date components, pressure difference, dew point spread, and humidity lag. This is used at startup for the global dataset and implicitly in how new data rows are constructed during iterative forecasting.
* **Prediction Endpoint (`/predict`)**: The main endpoint for obtaining weather forecasts.

### 3. /predict Endpoint Logic

This endpoint handles `POST` requests to generate weather forecasts.

* **Request (`ForecastInput` Pydantic model)**:
    * `kecamatan_name` (string): The name of the sub-district.
    * `start_date` (string, format "YYYY-MM-DD"): The first day for the forecast.
    * `forecast_days` (integer, optional, default: 7): The number of days to forecast.
* **Processing Steps**:
    1.  **Validation**:
        * Checks if the `start_date` string is in the correct "YYYY-MM-DD" format.
        * Verifies that the `kecamatan_name` exists in the `df_global` and can be transformed by the loaded `label_encoder`.
    2.  **Historical Data Retrieval**:
        * Filters `df_global` for the specified `kecamatan_name` and sorts it by `datetime`.
        * Selects the last `TIME_STEPS` (7 days) of data ending on the day *before* the `start_date`. This forms the initial input sequence.
        * If fewer than `TIME_STEPS` days of historical data are found for the given `kecamatan` before the `start_date`, an error is raised.
    3.  **Kecamatan Encoding**: The input `kecamatan_name` is transformed into its numerical representation using the loaded `label_encoder`.
    4.  **Iterative Forecasting Loop (for `forecast_days`)**:
        
        a.  **Prepare Numerical Input**: The numerical features (from `EXPECTED_FEATURES_ORDER`) of the current `TIME_STEPS`-day sequence are extracted.
        
        b.  **Scale Features**: These numerical features are scaled using the `feature_scaler`. The sequence is reshaped to `(1, TIME_STEPS, n_features)`.
        
        c.  **Predict**: The scaled numerical sequence and the encoded `kecamatan` are passed to the `model.predict()` method.
        
        d.  **Inverse Transform Targets**: The model's scaled output predictions are inverse-transformed using the `target_scaler` to get the actual weather values (`precipprob`, `windspeed`, `temp`, `humidity`).
        
        e.  **Store Output**: The prediction for the current day (datetime, kecamatan, and the four target variables) is stored. Values are rounded: `precipprob` to 4 decimal places, others to 2.
        
        f.  **Prepare Next Input Sequence (if not the last forecast day)**:
            i.  The date for the next prediction step is determined (current forecast date).
            ii. A new row of features is constructed for this next day:
                * Datetime features (`year`, `month`, `day`, `weekday`) are updated based on the new date.
                * `humidity_lag1`: Set to the `humidity` predicted in the current step (d).
                * `dew`: Carried forward from the last known actual or estimated value in the sequence.
                * `dew_point_spread`: Calculated using the predicted `temp` (d) and the carried-forward `dew`.
                * `pressure`: Carried forward from the last known actual or estimated value.
                * `pressure_diff`: Calculated using the carried-forward `pressure` and the `pressure` from the row before it in the sequence.
                * Other features in `EXPECTED_FEATURES_ORDER` are carried over from the last row of the current sequence.
            iii. This newly constructed row is appended to the current sequence, and the oldest row in the sequence is dropped to maintain the `TIME_STEPS` window. The DataFrame columns are realigned if necessary.
    5.  **Response (`List[ForecastOutput]`)**: Returns a list of JSON objects, each representing the forecast for a day, structured according to the `ForecastOutput` Pydantic model.

* **Error Handling**: The endpoint uses `try-except` blocks and FastAPI's `HTTPException` to return appropriate HTTP status codes and error messages for issues like:
    * Invalid `start_date` format.
    * `kecamatan_name` not found in the historical data or label encoder.
    * Insufficient historical data to form the initial sequence.
    * Data shape mismatches or other processing errors during feature preparation.
    * Failures during model prediction or data transformation.
    * Other unexpected server-side errors.

---

## How to Use the API

### Prerequisites

1.  Ensure the FastAPI server is running. You can typically start it with a command like:
    ```bash
    uvicorn main:app --reload
    ```
    (Assuming your FastAPI application instance is named `app` in a file named `main.py`).
2.  The API will be available at a local address, usually `http://127.0.0.1:8000`.

### 1. Home Endpoint (`/`)

* **URL**: `/`
* **Method**: `GET`
* **Description**: A simple endpoint to check if the API is running.
* **Response**:
    ```json
    {
      "message": "Weather Forecast API. Use the /predict endpoint to get forecasts."
    }
    ```

### 2. Predict Endpoint

* **URL**: `/predict`
* **Method**: `POST`
* **Description**: Submits data to get a weather forecast for a specified `kecamatan` and date range.
* **Request Body (JSON - `ForecastInput`)**:
    ```json
    {
      "kecamatan_name": "string",
      "start_date": "YYYY-MM-DD",
      "forecast_days": integer
    }
    ```
    * `kecamatan_name`: The name of the sub-district (e.g., `"trikora"`). This must be a name present in the historical data and known to the model's label encoder.
    * `start_date`: The first date for which you want a forecast (e.g., `"2025-01-29"`).
    * `forecast_days` (optional): The number of days to forecast from the `start_date`. Defaults to 7 if not provided.

* **Example Request (using cURL)**:
    ```bash
    curl -X 'POST' \
      '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "kecamatan_name": "trikora",
        "start_date": "2025-01-29",
        "forecast_days": 3
      }'
    ```

* **Successful Response (JSON - `List[ForecastOutput]`)**:
    A list of forecast objects, one for each day.
    ```json
    [
      {
        "datetime": "2025-01-29",
        "kecamatan": "trikora",
        "precipprob": 0.6550,    // Example value (rounded to 4 decimal places)
        "windspeed": 15.25,      // Example value (rounded to 2 decimal places)
        "temp": 28.50,           // Example value (rounded to 2 decimal places)
        "humidity": 75.12        // Example value (rounded to 2 decimal places)
      },
      {
        "datetime": "2025-01-30",
        "kecamatan": "trikora",
        "precipprob": 0.6010,
        "windspeed": 14.80,
        "temp": 28.20,
        "humidity": 74.50
      },
      {
        "datetime": "2025-01-31",
        "kecamatan": "trikora",
        "precipprob": 0.5500,
        "windspeed": 14.00,
        "temp": 27.90,
        "humidity": 73.00
      }
    ]
    ```
    *(Actual values will vary based on the model's prediction and input data.)*

* **Error Responses**:
    * **400 Bad Request**: If input data is invalid.
        ```json
        {
          "detail": "Invalid start_date format. Please use YYYY-MM-DD."
        }
        ```
        ```json
        {
          "detail": "Kecamatan 'some_unknown_place' not seen during training. Cannot encode."
        }
        ```
        ```json
        {
          "detail": "Not enough historical data for kecamatan 'kecamatan_name' before YYYY-MM-DD. Need 7 days, found X."
        }
        ```
         ```json
        {
          "detail": "Input data processing error: [specific error message from ValueError]"
        }
        ```
    * **404 Not Found**: If historical data for the specified `kecamatan_name` doesn't exist.
        ```json
        {
          "detail": "No historical data found for kecamatan 'some_kecamatan'."
        }
        ```
    * **503 Service Unavailable**: If the model, preprocessors, or global data failed to load at startup, preventing the API from being ready. (This is a general indication; specific startup errors are logged to the console).
        ```json
        {
          "detail": "Model, preprocessors, or global data not loaded. API is not ready."
        }
        ```
    * **500 Internal Server Error**: For other unexpected errors during processing.
        ```json
        {
          "detail": "An unexpected error occurred: [specific error message]"
        }
        ```

You can also interact with the API through its auto-generated interactive documentation (Swagger UI) by navigating to `http://127.0.0.1:8000/docs` in your browser when the FastAPI server is running, or to `http://127.0.0.1:8000/redoc` for ReDoc documentation.