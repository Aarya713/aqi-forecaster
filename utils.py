
import joblib
import numpy as np
import pandas as pd
import requests
import openmeteo_requests
from datetime import datetime, timedelta, date
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Load model and scalers (from Google Drive)
# ============================================
model_folder = ''   # empty because the files are in the same folder

model = load_model(model_folder + 'best_aqi_model.h5', compile=False)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

scaler_X = joblib.load(model_folder + 'scaler_X.pkl')
scaler_y = joblib.load(model_folder + 'scaler_y.pkl')
selected_features = joblib.load(model_folder + 'selected_features.pkl')

print("✅ Model and scalers loaded successfully.")
print(f"Number of features expected: {len(selected_features)}")

# ============================================
# API Keys
# ============================================
WAQI_TOKEN = '9dddf2226d2d3f583ee414f44bb386f82b8886ef'
OPENA_KEY = '9655ef76cc890e4cbfe3eb8ac1a841ce2e73b1d328f31add5db5bcd3a2bccc32'
CITY = 'Delhi'

# ============================================
# Real‑time weather (Open-Meteo)
# ============================================
def fetch_weather_last_30_days():
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    om = openmeteo_requests.Client()
    params = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ["temperature_2m_max", "temperature_2m_min",
                  "apparent_temperature_max", "apparent_temperature_min",
                  "precipitation_sum", "rain_sum",
                  "weather_code", "wind_speed_10m_max", "wind_gusts_10m_max"]
    }
    responses = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    daily = responses[0].Daily()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')[:30]   # exactly 30 days
    data = []
    for i, d in enumerate(dates):
        data.append({
            'date': d,
            'temperature_2m_max': daily.Variables(0).Values(i),
            'temperature_2m_min': daily.Variables(1).Values(i),
            'apparent_temperature_max': daily.Variables(2).Values(i),
            'apparent_temperature_min': daily.Variables(3).Values(i),
            'precipitation_sum': daily.Variables(4).Values(i),
            'rain_sum': daily.Variables(5).Values(i),
            'weather_code': daily.Variables(6).Values(i),
            'wind_speed_10m_max': daily.Variables(7).Values(i),
            'wind_gusts_10m_max': daily.Variables(8).Values(i),
        })
    df = pd.DataFrame(data).set_index('date')
    df['wind_direction_10m_dominant'] = 0   # dummy
    return df

# ============================================
# Pollution data (WAQI for current pollutants, OpenAQ for historical AQI)
# ============================================
def fetch_pollution_last_30_days():
    import requests
    from datetime import date, timedelta

    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(end=end_date, periods=30, freq='D')

    # --- 1. Current pollutant values from WAQI (repeated over 30 days) ---
    url_waqi = f"https://api.waqi.info/feed/{CITY}/?token={WAQI_TOKEN}"
    resp_waqi = requests.get(url_waqi).json()
    if resp_waqi['status'] != 'ok':
        raise Exception(f"WAQI API error: {resp_waqi.get('data', 'Unknown')}")
    current = resp_waqi['data']['iaqi']
    aqi_today = resp_waqi['data']['aqi']

    # Build DataFrame with 30 identical rows of pollutants
    data = []
    for d in dates:
        row = {
            'date': d,
            'PM2.5': current.get('pm25', {}).get('v', np.nan),
            'PM10':  current.get('pm10', {}).get('v', np.nan),
            'NO2':   current.get('no2', {}).get('v', np.nan),
            'SO2':   current.get('so2', {}).get('v', np.nan),
            'CO':    current.get('co', {}).get('v', np.nan),
            'Ozone': current.get('o3', {}).get('v', np.nan),
        }
        data.append(row)
    df = pd.DataFrame(data).set_index('date')

    # --- 2. Historical PM2.5 from OpenAQ (used as AQI proxy) ---
    try:
        url_openaq = "https://api.openaq.org/v3/measurements"
        params = {
            "location": "Delhi",
            "parameter": "pm25",
            "date_from": start_date.isoformat(),
            "date_to": end_date.isoformat(),
            "limit": 1000
        }
        headers = {"X-API-Key": OPENA_KEY}
        resp_openaq = requests.get(url_openaq, params=params, headers=headers).json()
        if 'results' in resp_openaq and len(resp_openaq['results']) > 0:
            pm25_by_date = {}
            for rec in resp_openaq['results']:
                d = pd.to_datetime(rec['date']['utc']).date()
                if d not in pm25_by_date:
                    pm25_by_date[d] = []
                pm25_by_date[d].append(rec['value'])
            daily_pm25 = {d: np.mean(vals) for d, vals in pm25_by_date.items()}
            aqi_series = pd.Series(daily_pm25).reindex(dates, method='ffill')
            aqi_series.fillna(aqi_today, inplace=True)
            df['AQI'] = aqi_series
        else:
            # Fallback: use WAQI's current AQI for all days
            df['AQI'] = aqi_today
    except Exception as e:
        print(f"OpenAQ fetch failed: {e}. Falling back to flat trend.")
        df['AQI'] = aqi_today

    df.index = pd.to_datetime(df.index)
    return df

# ============================================
# Feature engineering (must match training)
# ============================================
def engineer_features(df):
    """Re‑create all features exactly as in training, with forward fill."""
    df.index = pd.to_datetime(df.index)
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Days'] = df.index.dayofweek + 1          # Monday=1 … Sunday=7
    df['Holidays_Count'] = 0

    if 'wind_direction_10m_dominant' in df.columns:
        df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction_10m_dominant']))
        df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction_10m_dominant']))

    rain_threshold = 0.1
    df['rain_event'] = (df['precipitation_sum'] > rain_threshold).astype(int)
    df['days_since_rain'] = df.groupby(df['rain_event'].cumsum()).cumcount()

    df['is_weekend'] = (df['Days'].isin([6,7])).astype(int)

    month_dummies = pd.get_dummies(df['Month'], prefix='month')
    for m in range(1, 13):
        col = f'month_{m}'
        if col not in month_dummies.columns:
            month_dummies[col] = 0
    df = pd.concat([df, month_dummies], axis=1)

    for lag in [1, 2, 3, 7]:
        df[f'AQI_lag_{lag}'] = df['AQI'].shift(lag)

    for window in [3, 7, 30]:
        df[f'AQI_roll_mean_{window}'] = df['AQI'].rolling(window).mean()
        df[f'AQI_roll_std_{window}'] = df['AQI'].rolling(window).std()

    alpha = 0.3
    df['PM25_ewma'] = df['PM2.5'].ewm(alpha=alpha, adjust=False).mean()

    df['rain_wind_interaction'] = df['precipitation_sum'] * df['wind_speed_10m_max']

    df['PM25_PM10_ratio'] = df['PM2.5'] / (df['PM10'] + 1e-6)
    df['NO2_SO2_ratio']   = df['NO2'] / (df['SO2'] + 1e-6)
    df['CO_NO2_ratio']    = df['CO'] / (df['NO2'] + 1e-6)

    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

# ============================================
# Internal helper to predict from a merged DataFrame
# ============================================
def _predict_from_df(df):
    """Internal prediction function that takes a merged dataframe (pollution + weather)."""
    if len(df) == 0:
        raise ValueError("Empty dataframe")
    # Ensure at least 30 rows (pad if needed)
    if len(df) < 30:
        first_row = df.iloc[[0]]
        extra = []
        for i in range(1, 30 - len(df) + 1):
            new_date = first_row.index[0] - timedelta(days=i)
            new_row = first_row.copy()
            new_row.index = [new_date]
            extra.append(new_row)
        extra.reverse()
        df = pd.concat(extra + [df])
        df = df.sort_index()
    # Add 29 synthetic days before earliest for rolling windows
    earliest = df.index.min()
    synth = []
    for i in range(1, 30):
        new_date = earliest - timedelta(days=i)
        new_row = df.iloc[[0]].copy()
        new_row.index = [new_date]
        synth.append(new_row)
    synth.reverse()
    extended_df = pd.concat(synth + [df])
    extended_df = extended_df.sort_index()
    # Feature engineering
    df_feat = engineer_features(extended_df)
    if len(df_feat) < 30:
        raise ValueError(f"Only {len(df_feat)} rows after engineering – need 30.")
    df_last30 = df_feat.iloc[-30:]
    X = df_last30[selected_features].values
    X_scaled = scaler_X.transform(X)
    X_seq = X_scaled.reshape(1, 30, -1)
    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
    return y_pred

# ============================================
# Public prediction function
# ============================================
def predict_next_day():
    weather_df = fetch_weather_last_30_days()
    poll_df = fetch_pollution_last_30_days()
    df = poll_df.join(weather_df, how='inner')
    if len(df) == 0:
        raise ValueError("No overlapping dates")
    return _predict_from_df(df)

# ============================================
# What‑if simulation
# ============================================
def simulate_aqi(changes):
    """changes: dict like {'PM2.5': 0.9, 'NO2': 0.8} (reduction factor)"""
    weather_df = fetch_weather_last_30_days()
    poll_df = fetch_pollution_last_30_days()
    # Apply changes to the whole 30‑day dataframe
    for col, factor in changes.items():
        if col in poll_df.columns:
            poll_df[col] = poll_df[col] * factor
    df = poll_df.join(weather_df, how='inner')
    if len(df) == 0:
        raise ValueError("No overlapping dates")
    return _predict_from_df(df)
