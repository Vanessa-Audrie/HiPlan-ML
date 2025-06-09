# HiPlan - Machine Learning Documentation

HiPlan-ML is a repository specifically use to store machine learning models used in [HiPlan](https://github.com/HarHamz/HiPlan) to assist hikers with:
- Weather forecasting based on historical time-series data.
- Difficulty and time prediction based on mountain features and weather.
- Recommender system to suggest mountains based on user preferences.
- A Streamlit app for simple model inference and interaction.

This repository contains multiple models and experiments, each stored in separate folders with their own documentation. 

For the full project, check [HiPlan](https://github.com/HarHamz/HiPlan) repository.

---

## Table of Contents
1.  [Project Structure](#📁-project-structure)
2.  [Data ETL](#data-etl)
3.  [Weather Prediction](#weather-prediction)
4.  [Difficulty and Time Prediction](#difficulty-and-time-prediction)
5.  [Recommender System](#recommender-system)
6.  [Simple Streamlit Inference](#simple-streamlit-inference)
---

## 📁 Project Structure

```bash
HiPlan-ML/
├── data etl/                               # data collecting and cleaning folder
├── weather-prediction-model/               # monthly weather prediction (model 1) 
├── difficulty_and_time_prediction_model/   # difficulty and time prediction model (model 2)
├── recommender-system/                     # recommender system (content based)
├── requirements.txt                        # requirements ONLY for streamlit inference
├── streamlit_inference.py                  # a simple inference deployed in streamlit
└── README.md                               # main ReadMe file
```
For more detailed information, please access each folder ReadMe file 

## Data ETL
This folder contains scripts for Extract, Transform, Load (ETL) processes. Raw hiking-related datasets are cleaned, transformed, and structured to support various models in the project. This includes operations such as handling missing values and preparing data for training and inference.

## Monthly Weather Prediction
The Monthly Weather Prediction module is designed to forecast critical weather parameters that influence hiking safety and experience. These include precipitation probability (precipprob), wind speed (windspeed), temperature (temp), and humidity (humidity). Weather predictions provide users with insights into upcoming conditions, so they can plan their hikes more effectively.

## Difficulty and Time Prediction
The Difficulty and Time Prediction module estimates both the difficulty level and the expected duration of a hike. It utilizes key features such as mountain height, trail distance, and elevation gain, along with real-time or forecasted weather conditions like precipitation probability, wind speed, temperature, and humidity. This helps users evaluate whether a trail aligns with their physical capabilities and available time.

## Recommender System
The Recommender System uses a content-based filtering approach to recommend mountains that are similar to the user's preferences. Based on user-provided input such as location and mountain elevation, the system compares these inputs to the available mountain profiles and suggests similar options. This method allows for personalized recommendations even without prior user history, making it ideal for first-time users or those exploring new hiking areas.

## Simple Streamlit Inference
The Simple Streamlit Inference module provides a user-friendly web interface for interacting with all the models. Through this application, users can input relevant parameters and instantly receive predictions on weather, difficulty, time estimation, and hiking recommendations. This integration ensures that the entire system is accessible and usable by non-technical users in a seamless and interactive manner.

### To run the streamlit inference:
Install requirements:
```
pip install -r requirements.txt
```

Run it locally:
```
streamlit run streamlit_inference.py
```

Or use this deployed link to view it directly [Streamlit Inference](https://hiplan-ml-inference.streamlit.app/)