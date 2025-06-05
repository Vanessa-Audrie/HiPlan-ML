# Weather Forecasting Model and API Documentation

This document provides an overview of the weather forecasting model, the API built to serve it, and instructions on how to use the API. Made for HiPlan.

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

* **Dataset**: The model is trained on a time-series weather dataset (`weather.csv`) containing historical weather observations.
* **Feature Engineering**:
    * **Datetime Features**: `year`, `month`, `day`, `weekday` are extracted from the `datetime` column.
    * **Differential Features**: `pressure_diff` (change in pressure) and `dew_point_spread` (difference between temperature and dew point) are calculated.
    * **Lagged Features**: `humidity_lag1` (humidity from the previous day) is created.
    * **Categorical Feature**: `kecamatan` (sub-district/area) is a key categorical input.
* **Target Variables**: `precipprob`, `windspeed`, `temp`, `humidity`.

### 2. Preprocessing

* **Cleaning**: Missing values (`dropna`) and duplicates (`drop_duplicates`) are removed.
* **Encoding**: The `kecamatan` categorical feature is converted into numerical representation using `LabelEncoder`.
* **Scaling**: Numerical features are scaled using `RobustScaler` to handle outliers effectively. Target variables are also scaled before training and inverse-transformed after prediction.
* **Sequencing**: The time-series data is transformed into sequences. The model uses a sequence of the past **7 days** (`TIME_STEPS = 7`) of numerical features to predict the weather for the next day.

### 3. Model Architecture

The model is a **Neural Network** built with TensorFlow/Keras, specifically designed for sequence data:

* **Inputs**:
    1.  `numerical_input`: A sequence of scaled numerical features for the past `TIME_STEPS` days. (Shape: `(TIME_STEPS, n_features)`)
    2.  `kecamatan_input`: The encoded `kecamatan` for which the prediction is being made. (Shape: `(1,)`)
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

* The data is split into training and testing sets based on the year (data up to 2023 for training, after 2023 for testing).
* **Early Stopping**: Used during training to prevent overfitting by monitoring `val_loss` with a `patience` of 10.


---

## API Explanation

The API is built using **FastAPI** to serve the trained weather forecasting model.

### 1. Startup

`@app.on_event("startup")`

When the API server starts:
* **Model Loading**: The pre-trained Keras model (`weather_prediction_lstm_model.keras`) is loaded.
* **Preprocessor Loading**: Saved preprocessors (`weather_preprocessors.pkl`), which include the `feature_scaler`, `target_scaler`, and `label_encoder`, are loaded using `pickle`.
* **Feature List Loading**: The specific list and order of features (`EXPECTED_FEATURES_ORDER`) used during training (`weather_feature_list.pkl`) are loaded.
* **Global Data Loading & Preprocessing**:
    * The historical weather data (`weather.csv`) is loaded into a global pandas DataFrame (`df_global`).
    * The `engineer_features` function is applied to this DataFrame to create necessary features.
    * Rows with NA values in critical feature columns are dropped.
    * This global DataFrame is used as the source of historical data when making new predictions.

### 2. Core Functionality

* **Feature Engineering (`engineer_features`)**: A utility function that takes a DataFrame and adds derived features like date components, pressure difference, dew point spread, and humidity lag. This is used both at startup for the global dataset and potentially for preparing input for new predictions if structured differently.
* **Prediction Endpoint (`/predict`)**: This is the main endpoint for getting weather forecasts.

### 3. /predict Endpoint Logic

* **Request**: Accepts a `POST` request with a JSON body containing:
    * `kecamatan_name` (string): The name of the sub-district.
    * `start_date` (string, "YYYY-MM-DD"): The first day for the forecast.
    * `forecast_days` (integer, optional, default: 7): The number of days to forecast.
* **Processing Steps**:
    1.  **Validation**: Checks if the model and preprocessors are loaded. Validates the `start_date` format.
    2.  **Historical Data Retrieval**:
        * Filters the `df_global` for the specified `kecamatan_name`.
        * Selects the last `TIME_STEPS` (7 days) of data ending on the day *before* the `start_date`. This forms the initial input sequence.
        * If insufficient historical data is found, an error is raised.
    3.  **Kecamatan Encoding**: The input `kecamatan_name` is transformed using the loaded `label_encoder`.
    4.  **Iterative Forecasting Loop (for `forecast_days`)**:
        a.  **Prepare Input**: The numerical features from the current sequence are extracted, ensuring they are in the `EXPECTED_FEATURES_ORDER`.
        b.  **Scale Features**: The numerical features are scaled using the `feature_scaler`.
        c.  **Predict**: The scaled numerical sequence and the encoded `kecamatan` are passed to the `model.predict()` method.
        d.  **Inverse Transform**: The scaled output predictions are inverse-transformed using the `target_scaler` to get the actual weather values.
        e.  **Store Output**: The prediction for the current day (datetime, kecamatan, and target variables) is stored.
        f.  **Prepare Next Input Sequence (if not the last forecast day)**:
            * The date is advanced by one day.
            * A new row of features is constructed for this next day:
                * Date features (`year`, `month`, `day`, `weekday`) are updated.
                * `humidity_lag1` for the new row is set to the `humidity` predicted in the current step.
                * `dew` is typically carried forward or estimated to calculate `dew_point_spread` with the newly predicted `temp`.
                * `pressure` is typically carried forward to calculate `pressure_diff`.
                * Other features in `EXPECTED_FEATURES_ORDER` are carried over from the last known row or set to a default if not directly updated.
            * This new row is appended to the current sequence, and the oldest row is dropped to maintain the `TIME_STEPS` window.
    5.  **Response**: Returns a list of JSON objects, each representing the forecast for a day.

* **Error Handling**: The endpoint includes `try-except` blocks to catch issues like `FileNotFoundError` (for model/preprocessors), `ValueError` (e.g., invalid date, kecamatan not found in encoder, data shape mismatch), and other general exceptions, returning appropriate HTTP status codes and error messages.

---

## How to Use the API

### Prerequisites

1.  Ensure the FastAPI server is running. You can typically start it with a command like:
    ```bash
    uvicorn main:app --reload
    ```
    (Assuming your FastAPI application instance is named `app` in a file named `main.py`).
2.  The API will be available at a local address, usually `http://127.0.0.1:8000`.

### 1. Home Endpoint

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
* **Description**: Submits data to get a weather forecast.
* **Request Body (JSON)**:
    ```json
    {
      "kecamatan_name": "string",
      "start_date": "YYYY-MM-DD",
      "forecast_days": integer 
    }
    ```
    * `kecamatan_name`: The name of the sub-district (e.g., `"trikora"`). This must be a name the model was trained on or has been added to its label encoder.
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

* **Successful Response (JSON)**:
    A list of forecast objects, one for each day.
    ```json
    [
      {
        "datetime": "2025-01-29",
        "kecamatan": "trikora",
        "precipprob": 65.5,     // Example value
        "windspeed": 15.25,     // Example value
        "temp": 28.5,           // Example value
        "humidity": 75.12       // Example value
      },
      {
        "datetime": "2025-01-30",
        "kecamatan": "trikora",
        "precipprob": 60.1,
        "windspeed": 14.8,
        "temp": 28.2,
        "humidity": 74.5
      },
      {
        "datetime": "2025-01-31",
        "kecamatan": "trikora",
        "precipprob": 55.0,
        "windspeed": 14.0,
        "temp": 27.9,
        "humidity": 73.0
      }
    ]
    ```
    *(Values are examples and will vary based on the model's actual prediction)*

* **Error Responses**:
    * **400 Bad Request**: If input data is invalid (e.g., wrong date format, `kecamatan_name` not known, not enough historical data for the requested `start_date` and `kecamatan_name`).
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
    * **404 Not Found**: If historical data for the specified `kecamatan_name` doesn't exist in the loaded `df_global`.
        ```json
        {
          "detail": "No historical data found for kecamatan 'some_kecamatan'."
        }
        ```
    * **503 Service Unavailable**: If the model or preprocessors failed to load at startup.
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

You can also interact with the API through its auto-generated documentation if you navigate to `http://127.0.0.1:8000/docs` in your browser when the FastAPI server is running.