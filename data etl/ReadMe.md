## Dataset

The dataset results can be seen in [Dataset](https://drive.google.com/drive/folders/1McI8nlHV-b66qySo-BVqkQ69yds2_u57?usp=sharing)

Mountain Dataset Information
| Column           | Data Type | Description                                                                              |
| ---------------- | --------- | -----------------------------------------------------------------------------------------|
| `nama`           | `object`  | Mountain name                                                                            |
| `provinsi`       | `object`  | Province where the mountain is located                                                   |
| `kabupaten`      | `object`  | The regency/city where the mountain is located.                                          |
| `kecamatan`      | `object`  | The district where the mountain is located                                               |
| `ketinggian`     | `int64`   | Height of mountain (in meters)                                                           |
| `jenis gunung`   | `object`  | Mountain types based on shape or geological origin (e.g. stratovolcano)                  |
| `status`         | `object`  | Volcanic activity status (active or inactive)                                            |
| `akses`          | `object`  | Information regarding the mountain's open and closed access status.                      |
| `jarak`          | `int64`   | Climbing distance from starting point to summit (in meters)                              |
| `elevation gain` | `int64`   | Total elevation gain during the hike (in meters).                                        |
| `latitude`       | `float64` | Mountain latitude coordinates (for geographic mapping)                                   |
| `longitude`      | `float64` | Mountain longtitude coordinates (for geographic mapping)                                 |

<br>

Weather Dataset Information
| Column           | Data Type        | Description                                                               |
| ---------------- | ---------------- | ------------------------------------------------------------------------- |
| `datetime`       | `datetime64[ns]` | Weather observation date in `YYYY-MM-DD` format                           |
| `datetimeEpoch`  | `int64`          | Date in epoch timestamp format (seconds since 1970-01-01)                 |
| `tempmax`        | `float64`        | Daily maximum temperature                                                 |
| `tempmin`        | `float64`        | Daily minimum temperature                                                 |
| `temp`           | `float64`        | Daily average temperature                                                 |
| `feelslikemax`   | `float64`        | Maximum perceived temperature                                             |
| `feelslikemin`   | `float64`        | Minimum perceived temperature                                             |
| `feelslike`      | `float64`        | Average perceived temperature                                             |
| `dew`            | `float64`        | Dew point, the temperature at which moisture begins to condense           |
| `humidity`       | `float64`        | Average humidity                                                          |
| `precip`         | `float64`        | Precipitation                                                             |
| `precipprob`     | `float64`        | Probability of precipitation                                              |
| `precipcover`    | `float64`        | Percentage of area affected by rain                                       |
| `snow`           | `float64`        | Snowfall                                                                  |
| `snowdepth`      | `float64`        | Snow depth                                                                |
| `windgust`       | `float64`        | Maximum wind gust speed                                                   |
| `windspeed`      | `float64`        | Average wind speed                                                        |
| `winddir`        | `float64`        | Wind direction                                                            |
| `pressure`       | `float64`        | Air pressure                                                              |
| `cloudcover`     | `float64`        | Cloud cover percentage                                                    |
| `visibility`     | `float64`        | Visibility distance                                                       |
| `solarradiation` | `float64`        | Solar radiation intensity                                                 |
| `solarenergy`    | `float64`        | Daily solar energy                                                        |
| `uvindex`        | `float64`        | Ultraviolet (UV) index                                                    |
| `sunrise`        | `object`         | Time of sunrise                                                           |
| `sunriseEpoch`   | `int64`          | Sunrise time in epoch format                                              |
| `sunset`         | `object`         | Time of sunset                                                            |
| `sunsetEpoch`    | `int64`          | Sunset time in epoch format                                               |
| `moonphase`      | `float64`        | Moon phase                                                                |
| `conditions`     | `object`         | General weather conditions (e.g., `rain, partially cloudy`)               |
| `description`    | `object`         | Detailed weather description (e.g., `partly cloudy throughout the day`)   |
| `icon`           | `object`         | Weather icon representation                                               |
| `stations`       | `object`         | Weather observation stations                                              |
| `source`         | `object`         | Data source (e.g., obs for observed data)                                 |
| `kecamatan`      | `object`         | Name of the district where the data was collected                         |
| `severerisk`     | `float64`        | Risk of extreme weather                                                   |
| `day`            | `int32`          | Day of the month (`1–31`)                                                 |
| `month`          | `int32`          | Month (`1–12`)                                                            |
| `weekday`        | `int32`          | Day of the week (`0=Monday, 6=Sunday`)                                    |
| `year`           | `int32`          | Year (`YYYY`)                                                             |

<br>

Difficulty and Estimation Time Dataset Information
| Column             | Data Type   | Description                           |
| ------------------ | ------------| --------------------------------------|
| `ketinggian`       | `int64`     | Height of mountain (m)                |
| `jarak`            | `int64`     | Distance from base to summit (m)      |
| `elevation_gain`   | `int64`     | Total elevation gain (mdpl)           |
| `temp`             | `float64`   | Daily average temperature (celsius)   |
| `humidity`         | `float64`   | Average humidity (%)                  |
| `precipprob`       | `float64`   | Probability of precipitation (%)      |
| `windspeed`        | `float64`   | Average wind speed (km/jam)           |
| `difficulty_score` | `float64`   | Trail Difficulty Score                |
| `estimated_time`   | `float64`   | Estimated Hiking Time                 |

<br>

# ETL Process
## Weather
### 1. Extract & Load (Raw)
Each ML member scrapes weather data from the [Visual Crossing](https://www.visualcrossing.com/) website for a different years, using the district (`kecamatan`) as the queried location (the list of district is taken from the mountain dataset). The scraping is divided based on the year:
- Vanessa: 2020, 2023
- Harry: 2024, 2025
- Faiza: 2021, 2022

Steps:
1. Open `scrape_weather_data.py`
2. Set:
   - Set the `YEAR` input according to the assigned range
   - Set the start and end dates in `START_DATE` and `END_DATE`
   - Input the API keys in `API_KEYS`
3. Run:
```bash
python scrape_weather_data.py
```

The result will be saved in the folder `YYYY/weather_per_region/` based on each district, with the filename format `weather_{kecamatan}`

### 2. Merge & Clean
Steps:
1. Merge all scraped data
2. Handle null values:
    - Empty `datetime` values are filled using the `wdatetime` column (some files use `wdatetime` instead of `datetime`, but they serve the same purpose)
    - Columns with a lot of null data but without important information will be dropped
    - Important columns with null values are filled using forward fill and backward fill.
3. Extract `day`, `month`, `weekday`, and `year` to clarify the date information.
4. Convert all data to lowercase to ensure consistency
5. Save the final file as a CSV named `merged_weather`

## Mountain
### 1. Extract 
Data is taken manually because not every website has complete data

### 2. Clean
Steps:
1. Remove redundant columns, such as distance being recorded in both kilometers and meters (the km column is removed since other data uses metric units in meters).
2. Remove units like (m) and (mdpl) from column names
3. Convert all data to lowercase to ensure consistency 
4. Parse coordinates into two separate columns: `latitude` and `longitude`
5. Save the file as a CSV named `gunung_indonesia`

## Model 2 Dataset
### 1. Extract & Load
Data is extracted by merging weather and mountain datasets with `on='kecamatan', how='inner'` 

### 2. Clean
Steps:
1. Select the features that will be used to create a new dataset (`features = ['ketinggian', 'jarak', 'elevation gain', 'temp', 'humidity', 'precipprob', 'windspeed']`)
2. Check the null and duplicate values
3. Drop duplicate values
4. Scale required column `'temp', 'humidity', 'precipprob', 'windspeed'` using `MinMaxScaler`

### 3. Create Dummy Targets
#### 1. Difficulty Score
```python
def base_difficulty(row):
    elevation_gain_ft = row['elevation gain'] * 3.28084  # konversi meter ke feet
    distance_mi = row['jarak'] / 1609.34  # konversi meter ke mile
    raw_score = math.sqrt(elevation_gain_ft * 2 * distance_mi) #shenandoah's formula
    return raw_score ** 0.5

def full_difficulty_score(row):
    base = base_difficulty(row)
    weather_penalty = (
        0.25 * row['precipprob_scaled'] +
        0.25 * row['windspeed_scaled'] +
        0.25 * row['temp_scaled'] +
        0.25 * row['humidity_scaled']
    )
    SCALE_FACTOR = 0.25
    final_score = base * weather_penalty * SCALE_FACTOR
    return final_score

# Normalize difficulty to 1–10 range
def normalize_difficulty(scores):
    min_score = scores.min()
    max_score = scores.max()
    return 1 + 9 * ((scores - min_score) / (max_score - min_score))
```
Base Difficulty is calculated using [Shenandoah's Hiking Difficulty Formula](https://www.nps.gov/shen/planyourvisit/how-to-determine-hiking-difficulty.htm), then adjusted with a weather penalty (the model doesn't only predict based on elevation gain and distance, but also considers the estimated weather on that day). The resulting score is then normalized to a scale of 1 to 10.

#### 2. Estimated Time

```python
def estimate_time(row):
    base_time = (row['jarak'] / 5000) + (row['elevation gain'] / 600) # naismith rule
    difficulty_factor = row['difficulty_score'] / 10 # scale 0–1
    return base_time * (1 + difficulty_factor)
```
Base Time is calculated using [Naismith’s Rule](https://www.restless-viking.com/2018/11/29/naismiths-rule/) where every 5,000 meters of distance equals 1 hour, plus an additional 1 hour for every 600 meters of elevation gain. This is then adjusted by adding the difficulty score to provide a more dynamic estimate that reflects the trail’s overall difficulty.

### 4. Save
Steps:
1. Drop unused columns `'temp_scaled', 'precipprob_scaled', 'windspeed_scaled', 'humidity_scaled', 'base_difficulty_score', 'full_difficulty_score'`
2. Rename the column `elevation gain` to `elevation_gain` for consistency
3. Save the file as a CSV named `model2_dataset`