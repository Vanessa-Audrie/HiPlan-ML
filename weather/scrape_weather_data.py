import pandas as pd
import requests
from time import sleep
import os

# Load kecamatan list
df_kecamatan = pd.read_csv("kecamatan.csv")
kecamatan_list = df_kecamatan['Kecamatan'].dropna().unique()

# API configuration
API_KEYS = [
    # 'visual_crossing_api_key1',
    # 'visual_crossing_api_key2',
    # 'etc',
]

key_index = 0
BASE_URL = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline'
START_DATE = 'yyyy-mm-dd' # Input start of the date that will be scraped
END_DATE = 'yyyy-mm-dd' # Input end of the date that will be scraped

# Create output folder
output_dir = "yyyy/weather_per_region" # Change yyyy to the year scraped
os.makedirs(output_dir, exist_ok=True)

# Initialize 429 error counter
rate_limit_errors = 0
MAX_429_ERRORS = 3

# Loop through kecamatan list
for kec in kecamatan_list:
    safe_name = kec.replace(" ", "").replace("/", "")
    filename = os.path.join(output_dir, f"weather_{safe_name}.csv")

    # Skip if already downloaded
    if os.path.exists(filename):
        print(f"[SKIP] Already exists: {filename}")
        continue 

    # Get Data
    print(f"[FETCHING] {kec}...")
    url = f"{BASE_URL}/{kec}/{START_DATE}/{END_DATE}"
    api_key = API_KEYS[key_index]
    params = {
        'unitGroup': 'metric',
        'include': 'days',
        'key': api_key,
        'contentType': 'json'
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 429:
            print(f"[RATE LIMIT] 429 Error with key {api_key}")

            # Move to the next API key
            key_index += 1
            if key_index >= len(API_KEYS):
                print("[ABORTING] All API keys exhausted.")
                break  # Exit script

            # Retry with the next key on next loop
            print(f"[SWITCHING] Trying next API key: {API_KEYS[key_index]}")
            sleep(2)
            continue  

        response.raise_for_status()
        data = response.json()

        days = data.get("days", [])
        if days:
            df = pd.DataFrame(days)
            df['Kecamatan'] = kec
            df.to_csv(filename, index=False)
            print(f"[SAVED] {filename}")
        else:
            print(f"[NO DATA] {kec}")

    except Exception as e:
        print(f"[ERROR] {kec}: {e}")

    sleep(1)  # Respect API rate limits