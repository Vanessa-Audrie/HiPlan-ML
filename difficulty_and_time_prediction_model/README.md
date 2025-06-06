# Difficulty and Time Estimation Model and API Documentation

This document provides an overview of the Difficulty and Time Estimation Model, the FastAPI-based API built to serve its predictions, and instructions on how to use the API. This project is intended for HiPlan.

---

## Table of Contents
1.  [Model Explanation](#model-explanation)
    1.  [Data and Features](#1-data-and-features)
    2.  [EDA](#2-eda)
    3.  [Preprocessing](#3-preprocessing)
    4.  [Model Architecture](#4-model-architecture)
    5.  [Evaluation](#5-evaluation)
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

The model predicts difficulty score and time estimation using the mountains data and weather conditions, specifically **height (`ketinggian`)**, **distance (`jarak`)**, **elevation gain (`elevation_gain`)**, **precipitation probability (`precipprob`)**, **wind speed (`windspeed`)**, **temperature (`temp`)**, and **humidity (`humidity`)**.

### 1. Data and Features

* **Dataset**: The model is trained on a dummy rule based [dataset](https://drive.google.com/file/d/11ev-8TxeOo_K9ToC08OCnmOnhyxJY-r-/view?usp=sharing). 
* **Mountain Features**:
    - `ketinggian`: Height of the mountain.
    - `jarak`: Distance of the trail.
    - `elevation_gain`: Elevation gain along the trail.
* **Weather Features**:
    - `precipprob`: Probability of precipitation.
    - `windspeed`: Wind speed.
    - `temp`: Temperature.
    - `humidity`: Humidity level.
<br>These features are combined to form the input for the model.
* **Target Variables**: `difficulty_score`, `estimated_time`.

### 2. EDA

* **Missing & Duplicate Values**: Checked for missing values using `df.isna().sum()` and identified any duplicate rows with `df.duplicated().sum()`. This ensures data quality before further processing.
* **Data Overview**: Used `df.info()` to inspect column types and non-null counts, helping to understand the structure of the dataset.
* **Feature Distributions**: Plotted histograms for all numerical features using `df.hist()` to visualize their distributions. This helps detect skewness, outliers, and patterns in the data.
* **Correlation Matrix**: Generated a heatmap of feature correlations using `sns.heatmap()` to identify strong positive or negative relationships between variables. This aids in feature selection and understanding multicollinearity.

### 3. Preprocessing

* **Split Data**: Data are split into 80:20, 80% of training data and 20% of testing data  
* **Scaling**: The features are scaled using **MinMaxScaler**, which transforms all values to a range between 0 and 1. This ensures that each feature contributes equally to the model training and avoids bias toward features with larger numerical ranges. The scaler is fitted **only** to the training data (`X_train`) and then applied to both the training and testing sets to maintain consistency.

### 4. Model Architecture

* **Model Architecture**:  
  A simple feedforward **Neural Network** was built using TensorFlow/Keras to predict two continuous outputs: **difficulty score** and **estimated time**. The architecture consists of:
  - An input layer matching the number of features
  - Two hidden `Dense` layers with ReLU activation (128 and 64 units)
  - An output layer with 2 units (for the two target variables)

* **Compilation**:  
  The model is compiled using:
  - Optimizer: `adam`
  - Loss Function: `Mean Squared Error (MSE)`
  - Metrics: `Mean Absolute Error (MAE)` and a custom-defined `Mean Absolute Percentage Error (MAPE)`

* **Callbacks**:
  - `ModelCheckpoint`: Saves the best model based on validation loss during training.
  - `EarlyStopping`: Stops training early if validation loss does not improve after 5 epochs and restores the best weights.

* **Training**:
  - The model is trained for up to 10 epochs using a batch size of 32.
  - Validation is performed using the 20% test split.

### 5. Evaluation
* **Custom Evaluation Metrics**:
  - `mean_absolute_percentage_error`: Calculates the average percentage error between predictions and actual values, ensuring division by zero is avoided using a small epsilon value.
  - `adjusted_r2_score`: Computes the adjusted R² score to assess model performance while considering the number of input features.

* **Evaluation & Visualization**:
  - After training, predictions are made on the test set.
  - A dual-plot visualizes:
    - The training and validation loss over epochs
    - The final **R²** and **MAPE (%)** scores for each target variable
  - These visualizations help assess the learning progress and prediction accuracy per output.

---

# Difficulty & Time Estimation API

This FastAPI-based service predicts the estimated hiking difficulty and duration based on mountains data and weather-related features. It loads a trained TensorFlow Keras model, along with a feature scaler and a predefined feature list, to ensure proper input formatting and normalization.

---

## API Explanation

### 1. Startup

On startup, the application performs the following:

- Loads a list of expected input features from `saved_model/feature_list.json`.
- Loads a fitted `MinMaxScaler` object from `saved_model/minmax_scaler.pkl`.
- Loads a trained Keras model from `saved_model/best_model.keras`.

It also enables **CORS** to allow access from frontend clients.

All necessary files must be present in the `saved_model/` directory:
- `best_model.keras`
- `minmax_scaler.pkl`
- `feature_list.json`
- `utils.py`

Logging is enabled to provide runtime diagnostics.

---

### 2. Core Functionality

The core functionality of the API is a POST endpoint at `/predict`, which:
- Accepts a JSON payload of mountain data and weather data.
- Formats and orders the input based on the loaded `feature_list.json`.
- Scales the input using `MinMaxScaler`.
- Performs a prediction using the loaded Keras model.
- Returns:
  - A difficulty score (rounded to 2 decimal places).
  - Estimated hiking time (in HH:MM format, using `hours_to_hh_mm` utility).

If the model output is not of shape `(1, 2)` which means the output is both difficulty score and estimated time, an error is raised.

---

### 3. /predict Endpoint Logic

- **Method**: `POST`
- **Endpoint**: `/predict`
- **Payload**:
  ```json
  {
    "ketinggian": float,
    "jarak": float,
    "elevation_gain": float,
    "temp": float,
    "precipprob": float,
    "windspeed": float,
    "humidity": float
  }


## How to Use the API


### Prerequisites

1.  Ensure the FastAPI server is running. You can typically start it with a command like:
    ```bash
    uvicorn main:app --reload
    ```
    (Make sure the FastAPI application instance is named `app` in a file named `main.py`).
2.  The API will be available at a local address, usually `http://127.0.0.1:8000`.


---


### 1. Home Endpoint

* **URL**: `/`
* **Method**: `GET`
* **Description**: A simple endpoint to verify if the API is active and running.
* **Response**:
    ```json
    {
      "message": "Difficulty & Time Estimation API"
    }
    ```

---

### 2. Predict Endpoint

* **URL**: `/predict`
* **Method**: `POST`
* **Description**: Predicts hiking difficulty score and estimated time from inputs of mountain data and weather features.

* **Request Body (JSON - `InputData`)**:
    ```json
    {
        "ketinggian": 4653,
        "jarak": 8000,
        "elevation_gain": 3674,
        "temp": 21,
        "precipprob": 90,
        "windspeed": 11.4,
        "humidity": 88
    }
    ```

* **Example Request (using cURL)**:
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "ketinggian": 4653,
        "jarak": 8000,
        "elevation_gain": 3674,
        "temp": 21,
        "precipprob": 90,
        "windspeed": 11.4,
        "humidity": 88
      }'
    ```

* **Successful Response**:
    ```json
    {
    "estimated_time": "12 jam 37 menit",
    "difficulty_score": 5.65
    }
    ```


* **Error Responses**:

    * **500 Internal Server Error** – Model or preprocessing error:
        ```json
        {
          "detail": "Prediction failed: [specific error message]"
        }
        ```

---

You can also interact with the API using Swagger UI by aading `/docs` to the local url:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc UI: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
