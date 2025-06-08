# Seasonal Weather Prediction Model and API

## 1. The Model: Predicting Seasonal Weather

The core of this model is a Deep Neural Network (DNN) designed to predict long-term, typical seasonal weather patterns for various locations (`kecamatan`).

### How it Works
#### Input Features
- **`kecamatan`**: The specific sub-district for the forecast. This is converted into a numerical format using One-Hot Encoding.
- **Year**: A simple numerical feature to capture long-term climate trends.
- **Cyclical Date Features**: To understand that December 31st is close to January 1st, the date is transformed into sine and cosine waves. This is a crucial step that allows the model to learn seasonal cycles effectively.
    - `day_sin` and `day_cos`: Represents the day of the year on a circular path.
    - `month_sin` and `month_cos`: Represents the month of the year on a circular path

#### Predicted Targets
The model is trained to predict four weather metrics simultaneously.
1. `precipprob`: The probability of precipitation (%)
1. `windspeed`: The average wind speed.
1. `temp`: The average temperature.
1. `humidity`: The average humidity. (%)

### Model Architecture
The model is a standard feed-forward Deep Neural Network built with TensorFlow/Keras. It consists of several `Dense` layers with `relu` activation and `Dropout` layers to prevent overfitting. It takes the preprocessed features as input and has an output layer with four neurons, one for each target variable.

## 2. The API: Predicting the Weather
The `main.py` file uses FastAPI to create a robust and easy-to-use web API for accessing the model's predictions.

### How to run the API
1. **Prerequisites**: Confirm the necessary libraries are installed.
```
    pip install -r requirements.txt
```
2. **File Placement**: Make sure the following files are in the same directory:
- `main.py`
- `weather_seasonal_model.keras` (The trained model)
- `weather_seasonal_preprocessor.pkl` (The data preprocessor)
- `requirements.txt`

3. **Start the Server**: Open a terminal in that directory and run the command:
```
    uvicorn main:app --reload
```
The server will start, and should be ac at http://127.0.0.1:8000.

### API Endpoints
Once running, the interactive API documentation can be accessed by navigating to http://127.0.0.1:8000/docs in a web browser.

### Status Check
- Endpoint: `/status`
- Description: A simple endpoint to confirm that the API is running and the model has been loaded successfully.
- Example URL: `http://127.0.0.1:8000/status`

### Daily Forecast for a Date Range
- Endpoint: /forecast/range
- Description: Provides a daily seasonal forecast for a specified number of consecutive days.
- Parameters:

    `kecamatan_name` (string): The name of the location (e.g., "Berastagi").

    `start_date` (string): The start date in `YYYY-MM-DD` format.

    `days_to_predict` (integer): The number of days to forecast (1-90).
- Example URL: `http://127.0.0.1:8000/forecast/range?kecamatan_name=Berastagi&start_date=2028-11-20&days_to_predict=5`

### Average Forecast for a Month
- Endpoint: /forecast/monthly
- Description: Calculates the average seasonal weather forecast for every day in a given month and year.
- Parameters:
    - kecamatan_name (string): The name of the location.
    - month (integer): The month number (1-12).
    - year (integer): The year.
- Example URL: `http://127.0.0.1:8000/forecast/monthly?kecamatan_name=Medan%20Sunggal&month=7&year=2030`

### Seasonality Analysis
- Endpoint: `/forecast/seasonality`
- Description: Determines if a given month is seasonally "Rainy" or "Sunny" based on the average monthly forecast.
- Logic: A month is classified as "Rainy" if one of the following conditions are met:
    - The average precipitation probability is > `70.0%`.
    - The average humidity is > `92.0%`.
    - Otherwise, it is classified as "Sunny".
- Parameters:
    - `kecamatan_name` (string): The name of the location.
    - `month` (integer): The month number (1-12).
    - `year` (integer): The year.
- Example URL: `http://127.0.0.1:8000/forecast/seasonality?kecamatan_name=Berastagi&month=11&year=2029`