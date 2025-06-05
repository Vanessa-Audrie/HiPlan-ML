# Weather Forecast API

## Table of Contents
1.  [Introduction](#introduction)
2.  [Prerequisites](#prerequisites)
3.  [Project Structure](#project-structure)
4.  [Setup](#setup)
5.  [Running the API](#running-the-api)
6.  [API Documentation (Swagger UI)](#api-documentation-swagger-ui)
7.  [Endpoints](#endpoints)
    * [GET / (Root)](#get--root)
    * [POST /predict (Get Weather Forecast)](#post-predict-get-weather-forecast)
8.  [Example Usage with cURL](#example-usage-with-curl)

## Introduction

This API provides weather forecasts including precipitation probability, wind speed, temperature, and humidity. It uses a deep learning (LSTM) model trained on historical weather data to make predictions for a specified `kecamatan` (district/sub-district) from a given start date.

The API loads historical data from `weather.csv` and uses it to construct the initial sequence for predictions, meaning you don't need to provide the historical sequence directly in your API calls.

## Prerequisites

Before you begin, ensure you have the following installed:
* Python (3.8 or newer recommended)
* pip (Python package installer)

You will also need the following files:
* `main.py`: The Python script containing the FastAPI application.
* `weather.csv`: The dataset containing historical weather data. This file **must** be present for the API to fetch historical data.
* `weather_prediction_lstm_model.keras`: The saved trained Keras model.
* `weather_preprocessors.pkl`: Saved Scikit-learn preprocessors (scalers, encoders).
* `weather_feature_list.pkl`: The list of feature names in the order expected by the model.

## Project Structure

For the API to run correctly, your project directory should ideally look like this:

```txt
api-directory/
├── main.py                             # Your FastAPI application code
├── weather.csv                         # Historical weather data
├── weather_prediction_lstm_model.keras # Trained Keras model
├── weather_preprocessors.pkl           # Saved preprocessors
├── weather_feature_list.pkl            # List of expected features
└── requirements.txt                    # Python dependencies
```

## Setup

1.  **Place Files**

    Ensure `main.py`, `weather.csv`, `weather_prediction_lstm_model.keras`, `weather_preprocessors.pkl`, and `weather_feature_list.pkl` are all in the same directory (e.g., `your-api-directory`).

2.  **Create `requirements.txt`**
    
    Create a file named `requirements.txt` in your project directory and add the following lines:
    ```txt
    fastapi==0.115.12
    numpy==2.2.6
    pandas==2.3.0
    pydantic==2.11.5
    tensorflow==2.19.0
    tensorflow_intel==2.16.1
    ```

3.  **Install Dependencies**
    
    Open your terminal or command prompt, navigate to your project directory, and run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the API

Once the setup is complete, you can start the API server.

1.  **Navigate to Directory:**
    In your terminal, make sure you are in the project directory where `main.py` is located.

2.  **Start Uvicorn Server:**
    Run the following command:
    ```bash
    uvicorn main:app --reload
    ```
    * `main`: Refers to the `main.py` file.
    * `app`: Refers to the FastAPI application instance named `app` inside `main.py`.
    * `--reload`: Enables auto-reloading when you make changes to the code (useful for development).

    You should see output similar to this, indicating the server is running:
    ```
    INFO:     Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000) (Press CTRL+C to quit)
    INFO:     Started reloader process [xxxxx] using statreload
    INFO:     Started server process [xxxxx]
    INFO:     Waiting for application startup.
    INFO:     Model loaded successfully.
    INFO:     Preprocessors loaded successfully.
    INFO:     Expected features order loaded: ['feature1', 'feature2', ...]
    INFO:     Global DataFrame loaded and preprocessed.
    INFO:     Global DataFrame shape after NA drop based on expected features: (YYYY, ZZ)
    INFO:     Application startup complete.
    ```
    The API will typically be available at `http://127.0.0.1:8000`.

## API Documentation (Swagger UI)

FastAPI automatically generates interactive API documentation. Once the server is running, you can access it in your web browser:

* **Swagger UI:** `http://127.0.0.1:8000/docs`
* **ReDoc:** `http://127.0.0.1:8000/redoc`

The Swagger UI (`/docs`) is particularly useful as it allows you to see all endpoints, their expected parameters, request bodies, and response schemas, and even try out the API directly from your browser.

## Endpoints

### GET `/` (Root)
* **Description:** Provides a welcome message for the API.
* **Method:** `GET`
* **Path:** `/`
* **Request Body:** None
* **Successful Response (200 OK):**
    ```json
    {
      "message": "Weather Forecast API. Use the /predict endpoint to get forecasts."
    }
    ```

### POST `/predict` (Get Weather Forecast)
* **Description:** Predicts weather conditions (precipitation probability, wind speed, temperature, humidity) for a specified `kecamatan_name`, starting from `start_date` for a given number of `forecast_days`. The API uses historical data from `weather.csv` for the given `kecamatan_name` up to the day before `start_date` to initialize the prediction sequence.
* **Method:** `POST`
* **Path:** `/predict`
* **Request Body (`application/json`):**
    | Field           | Type    | Description                                                                 | Required | Default |
    |-----------------|---------|-----------------------------------------------------------------------------|----------|---------|
    | `kecamatan_name`| string  | The name of the sub-district (must exist in `weather.csv` and training data). | Yes      |         |
    | `start_date`    | string  | The first date for the forecast, in "YYYY-MM-DD" format.                  | Yes      |         |
    | `forecast_days` | integer | The number of consecutive days to forecast.                                 | No       | 7       |

    **Example Request:**
    ```json
    {
      "kecamatan_name": "trikora",
      "start_date": "2025-01-29",
      "forecast_days": 3
    }
    ```

* **Successful Response (200 OK):**
    A JSON list of forecast objects, one for each forecasted day.
    ```json
    [
      {
        "datetime": "2025-01-29",
        "kecamatan": "trikora",
        "precipprob": 0.1234,
        "windspeed": 5.67,
        "temp": 28.9,
        "humidity": 75.5
      },
      {
        "datetime": "2025-01-30",
        "kecamatan": "trikora",
        "precipprob": 0.1500,
        "windspeed": 6.10,
        "temp": 29.1,
        "humidity": 76.0
      },
      {
        "datetime": "2025-01-31",
        "kecamatan": "trikora",
        "precipprob": 0.1450,
        "windspeed": 5.90,
        "temp": 29.0,
        "humidity": 75.8
      }
    ]
    ```
    *(Note: `precipprob`, `windspeed`, `temp`, `humidity` values are examples and will vary.)*

* **Possible Error Responses:**
    * **400 Bad Request:** If input data is invalid (e.g., malformed `start_date`, `kecamatan_name` not encoded during training, not enough historical data available in `weather.csv` before the `start_date` for the given `kecamatan_name`).
        ```json
        {
          "detail": "Not enough historical data for kecamatan 'example_kecamatan' before YYYY-MM-DD. Need 7 days, found X."
        }
        ```
        ```json
        {
          "detail": "Kecamatan 'unknown_kecamatan' not seen during training. Cannot encode."
        }
        ```
    * **404 Not Found:** If no historical data at all is found in `weather.csv` for the provided `kecamatan_name`.
        ```json
        {
          "detail": "No historical data found for kecamatan 'some_kecamatan_not_in_csv'."
        }
        ```
    * **503 Service Unavailable:** If the model, preprocessors, or global data failed to load during API startup. Check the server logs.
        ```json
        {
          "detail": "Model, preprocessors, or global data not loaded. API is not ready."
        }
        ```
    * **500 Internal Server Error:** If an unexpected error occurs on the server side. Check the server logs for more details.
        ```json
        {
          "detail": "An unexpected error occurred: <error_message>"
        }
        ```

## Example Usage with `cURL`

Here's how you can use `cURL` from your terminal to make a request to the `/predict` endpoint (ensure the API server is running):

```bash
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" \
-H "Content-Type: application/json" \
-d '{
  "kecamatan_name": "trikora",
  "start_date": "2024-07-15",
  "forecast_days": 2
}'