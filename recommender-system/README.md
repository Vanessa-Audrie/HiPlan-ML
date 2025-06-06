# Mountain System Recommendation and API Documentaion
This document outlines the Mountain Recommendation System, including the FastAPI-based API developed to serve personalized hiking suggestions, as well as detailed instructions on how to use the API. The system is designed as part of the HiPlan project to help users discover suitable hiking destinations based on their preferences.

## 📚 Table of Contents
1. [Model Explanation](#model-explanation)  
2. [Data and Features](#data-and-features)  
   1. [Dataset Source](#dataset-source)  
   2. [Selected Features](#selected-features)  
3. [Preprocessing & Model Preparation](#preprocessing--model-preparation)  
   1. [Saved Artifacts](#saved-artifacts)  
4. [Recommendation Pipeline](#recommendation-pipeline)  
5. [API Explanation](#api-explanation)  
   1. [Endpoints](#endpoints)  
   2. [Validation & Error Handling](#validation--error-handling)  
6. [Startup](#startup)  
   1. [Requirements](#requirements)  
   2. [Run the API](#run-the-api)  
7. [How to Use the API](#how-to-use-the-api)  
   1. [Test with curl or Postman](#test-with-curl-or-postman)  
   2. [Test Swagger UI](#test-swagger-ui)  


## Model Explanation
This system uses a **content-based filtering** approach to recommend mountains similar to user preferences. It combines location (text) and elevation (numerical) features to compute similarity using **cosine similarity**.

## Data and Features
This project uses a dataset containing information about various mountains in Indonesia to build a personalized hiking recommendation system.

### Dataset Source
Data is collected manually and contains detailed information on different mountains in indonesia, including their location, height, difficulty level, access, elevation gain, etc.

### Selected Features
The following features were selected for generating content-based recommendations:
- **Location**: User’s desired hiking area (used to match similar regions)
- **Elevation**: Used to find mountains with similar hiking altitude
These features are processed and used to compute similarity between mountains, helping the system provide more relevant and personalized hiking suggestions.

Each recommendation result includes the following information:
- **Mountain Name**: The recommended mountain
- **Province**: The province where the mountain is located.
- **Elevation**: The mountain’s elevation (mdpl)
- **Access**: Whether the mountain is currently open (Buka) or closed (Tutup) for hiking.

## Preprocessing & Model Preparation
The preprocessing and model preparation stage involves transforming and preparing the data so that it can be used for generating recommendations:

- ```TfidfVectorizer``` was used to convert mountain location text (e.g., province or region) into numerical vectors
- ```MinMaxScaler``` was used to normalize the mountain elevation data

These two feature sets (text and numeric) were combined using scipy.sparse.hstack to form a single feature matrix for similarity comparison

### Saved Artifacts
The resulting models and features were then fitted and saved as artifacts:

- ```vectorizer.pkl``` trained TF-IDF vectorizer
- ```scaler.pkl``` trained scaler for elevation
- ```combined_features.pkl```the final feature matrix for all mountains
- ```gunung_data.pkl``` the main dataset containing detailed mountain information as a serialized

## Recommendation Pipeline
This system uses a content-based filtering pipeline to recommend mountains that match the user's preferences. Rather than learning from historical user ratings (like in collaborative filtering), it compares input features directly with existing mountain data.

Steps in the Pipeline:

1. User Input
User provides two inputs, which are location and altitude

2. Vectorization & Scaling
The location is transformed using a pre-trained TfidfVectorizer to create a text-based feature vector, and the elevation is normalized using a pre-trained MinMaxScaler.

3. Feature Combination
The location vector and elevation value are combined using scipy.sparse.hstack() into one feature vector for similarity comparison.

4. Cosine Similarity Calculation
The user’s combined vector is compared to all existing mountain vectors using cosine similarity to measure closeness.

5. Filtering by Access Status
Only mountains marked as Akses = Buka (Open) are considered valid recommendations.

6. Top-N Selection
The system selects the top N most similar mountains based on the similarity score.

## API Explanation
This project includes a lightweight RESTful API built using **FastAPI** to serve the mountain recommendation model. The API allows users to submit location and elevation preferences, and it will respond with the top 5 mountain recommendations based on content-based similarity.

### Endpoints

#### `GET /`
**Response:**

```json
{ "message": "API Rekomendasi Gunung aktif" }
```

#### `POST /rekomendasi`
Returns top 5 mountains similar to user preference.

Request Body:
```json
{
  "lokasi": "Jawa Tengah",
  "ketinggian": 2500
}
```


Success Response:
```json
{
"rekomendasi": [
    {
      "Nama": "Gunung Merbabu",
      "Provinsi": "Jawa Tengah",
      "Ketinggian (dpl)": 3145,
      "Akses": "Buka"
    },
    ...
  ]
}
```

Error / Empty Result:
```json
{ "message": "⚠ Tidak ada gunung yang cocok ditemukan." }
```

### Validation & Error Handling
- ```lokasi``` must be a string
- ```ketinggian``` must be numeric
- ```400 errors``` for invalid input
- ```500 errors``` for internal failures


## Startup
### Requirements
Install required packages:
```
pip install -r requirements.txt
```

### Run the API
```
uvicorn main:app --reload
```
API will be available at: http://127.0.0.1:8000

## How to Use the API
### Test with curl or Postman
POST Request (example using curl):
```bash
curl -X POST "http://localhost:8000/rekomendasi" \
-H "Content-Type: application/json" \
-d '{"lokasi": "Jawa Barat", "ketinggian": 3000}'

```

### Test Swagger UI
Open your browser and navigate to:
```
http://localhost:8000/docs
```
