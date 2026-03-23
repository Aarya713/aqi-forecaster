# update_aqi.py
# Fetches today's AQI from WAQI and appends it to aqi_history.csv
# Designed to be run daily (e.g., via GitHub Actions)

import requests
import pandas as pd
from datetime import date

# WAQI token – you can hardcode it or read from environment variable
WAQI_TOKEN = "9dddf2226d2d3f583ee414f44bb386f82b8886ef"   # Replace with your actual token
CITY = "Delhi"

def fetch_today_aqi():
    url = f"https://api.waqi.info/feed/{CITY}/?token={WAQI_TOKEN}"
    resp = requests.get(url).json()
    if resp['status'] == 'ok':
        return resp['data']['aqi']
    else:
        raise Exception(f"WAQI API error: {resp.get('data', 'Unknown')}")

def main():
    today = date.today()
    aqi = fetch_today_aqi()

    # Try to load existing history, otherwise create a new DataFrame
    try:
        df = pd.read_csv('aqi_history.csv', index_col=0, parse_dates=True)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['AQI'])

    # Add today's value
    df.loc[today] = aqi

    # Keep only the last 90 days (optional)
    df = df.iloc[-90:]

    # Save back to CSV
    df.to_csv('aqi_history.csv')
    print(f"✅ Updated aqi_history.csv with {today}: AQI = {aqi}")

if __name__ == "__main__":
    main()
