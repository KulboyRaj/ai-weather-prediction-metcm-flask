import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import json
import requests
from datetime import datetime, timedelta

# Global variables
model = None
scalers = {}
feature_cols = []
df = None
lookback = 30
api_cache_file = 'weather_cache.json'
last_used_api = False
api_initialized = False


def initialize_model():
    """Initialize model and data once"""
    global model, scalers, feature_cols, df

    if model is None:
        try:
            print("🔄 Initializing weather model...")

            # Load dataset
            df = pd.read_csv('weather_cleaned.csv')
            df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

            feature_cols = [col for col in df.columns if col != 'datetime']

            # Scale features
            for col in feature_cols:
                scalers[col] = MinMaxScaler()
                scalers[col].fit(df[[col]])

            # Load model
            try:
                model = load_model('best_weather_lstm.h5', compile=False)
            except Exception as e1:
                print(f"⚠️ First load attempt failed: {e1}")
                try:
                    from tensorflow.keras.layers import InputLayer
                    def custom_input_layer(config):
                        if 'batch_shape' in config:
                            config['batch_input_shape'] = config['batch_shape']
                            del config['batch_shape']
                        return InputLayer(**config)

                    model = load_model('best_weather_lstm.h5', compile=False,
                                       custom_objects={'InputLayer': custom_input_layer})
                except Exception as e2:
                    print(f"❌ All loading attempts failed: {e2}")
                    raise

            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            raise


def initialize_api_cache():
    """Initialize API cache - try to fetch data for next 5 days via API"""
    global api_initialized

    if api_initialized:
        return

    print("🔄 Checking API cache...")

    # Calculate next 5 days from today
    today = datetime.now().date()
    next_5_days = [today + timedelta(days=i) for i in range(1, 6)]

    # Check cache for required dates
    cache_data = load_api_cache()
    missing_dates = []

    if cache_data and is_cache_fresh(cache_data):
        for target_date in next_5_days:
            if not has_data_for_date(cache_data, target_date):
                missing_dates.append(target_date.strftime('%Y-%m-%d'))

    if not missing_dates:
        print("✅ Cache has all required data for next 5 days")
        api_initialized = True
        return
    else:
        print(f"🔄 Cache missing data for: {', '.join(missing_dates)}")

    # Try direct API call to fetch missing data
    print("🌐 Trying API call to fetch missing weather data...")
    new_forecasts = fetch_weather_api()

    if new_forecasts:
        print("✅ API cache updated successfully")
        api_initialized = True
    else:
        print("⚠️ API call failed, will use available cache or model predictions")
        api_initialized = True


def fetch_weather_api(lat=18.5204, lon=73.8567):
    """Fetch weather data from OpenWeatherMap API"""
    global last_used_api
    try:
        API_KEY = "Add your api key here"
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

        print("🌐 Fetching weather data from API...")
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            forecasts = {}

            for item in data['list']:
                dt = datetime.fromtimestamp(item['dt'])
                date_str = dt.strftime('%Y-%m-%d')

                if date_str not in forecasts:
                    forecasts[date_str] = []

                forecast = {
                    'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'temp': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'sealevelpressure': item['main']['pressure'],
                    'windspeed': item['wind']['speed'] * 3.6,  # Convert m/s to km/h
                    'winddir': item['wind'].get('deg', 0),
                    'cloudcover': item['clouds']['all'],
                    'visibility': item.get('visibility', 10000) / 1000,  # Convert to km
                    'precip': item.get('rain', {}).get('3h', 0) or item.get('snow', {}).get('3h', 0),
                    'condition': item['weather'][0]['main']
                }

                # Calculate dew point
                temp = forecast['temp']
                humidity = forecast['humidity']
                forecast['dew'] = calculate_dew_point(temp, humidity)

                forecasts[date_str].append(forecast)

            # Merge with existing cache
            existing_cache = load_api_cache()
            if existing_cache and 'forecasts' in existing_cache:
                # Preserve existing data and add new forecasts
                existing_forecasts = existing_cache['forecasts']
                existing_forecasts.update(forecasts)  # New data overwrites old data for same dates
                forecasts = existing_forecasts

            # Save to cache
            save_api_cache(forecasts)
            print(f"✅ API data fetched for {len(forecasts)} days")
            last_used_api = True
            return forecasts
        else:
            print(f"❌ API request failed with status {response.status_code}")
            last_used_api = False
            return None

    except Exception as e:
        print(f"❌ API fetch failed: {e}")
        last_used_api = False
        return None


def calculate_dew_point(temp, humidity):
    """Calculate dew point from temperature and humidity"""
    # Magnus formula approximation
    alpha = 17.27
    beta = 237.7
    gamma = (alpha * temp) / (beta + temp) + np.log(humidity / 100.0)
    dew_point = (beta * gamma) / (alpha - gamma)
    return round(dew_point, 1)


def save_api_cache(forecasts):
    """Save API data to cache file with timestamp"""
    cache_data = {
        'last_updated': datetime.now().isoformat(),
        'forecasts': forecasts
    }
    with open(api_cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"💾 Cache saved with data for {len(forecasts)} days")


def load_api_cache():
    """Load API data from cache file"""
    try:
        if os.path.exists(api_cache_file):
            with open(api_cache_file, 'r') as f:
                cache_data = json.load(f)
            return cache_data
    except Exception as e:
        print(f"⚠️ Cache load failed: {e}")
    return None


def is_cache_fresh(cache_data, max_age_hours=6):
    """Check if cache is fresh (less than max_age_hours old)"""
    try:
        last_updated = datetime.fromisoformat(cache_data['last_updated'])
        cache_age = datetime.now() - last_updated
        return cache_age < timedelta(hours=max_age_hours)
    except:
        return False


def has_data_for_date(cache_data, target_date):
    """Check if cache has data for the target date"""
    if not cache_data or 'forecasts' not in cache_data:
        return False

    target_date_str = target_date.strftime('%Y-%m-%d')
    return target_date_str in cache_data['forecasts']

def get_api_prediction(target_date):
    """Get prediction from API cache if available, try API call if cache miss"""
    global last_used_api

    # Ensure target_date is a datetime object
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date, format='%Y-%m-%d', errors='coerce')

    target_date_str = target_date.strftime('%Y-%m-%d')

    # Load cached data
    cache_data = load_api_cache()

    # ✅ PRIORITY 1: Check if we have fresh cache data for the target date
    if cache_data and is_cache_fresh(cache_data) and has_data_for_date(cache_data, target_date):
        daily_data = cache_data['forecasts'][target_date_str]
        avg_data = calculate_daily_averages(daily_data)
        last_used_api = True
        print(f"✅ Using cached API data for {target_date_str}")
        return create_prediction_df(avg_data, target_date)

    # ❌ Cache miss - try direct API call
    print(f"🔄 No cached data for {target_date_str}, trying direct API call...")
    new_forecasts = fetch_weather_api()

    if new_forecasts and has_data_for_date({'forecasts': new_forecasts}, target_date):
        # ✅ PRIORITY 2: API success - use fresh data
        daily_data = new_forecasts[target_date_str]
        avg_data = calculate_daily_averages(daily_data)
        last_used_api = True
        print(f"✅ Using fresh API data for {target_date_str}")
        return create_prediction_df(avg_data, target_date)
    else:
        # ❌ API failed (network, API key, server down, etc.)
        last_used_api = False
        print(f"❌ API call failed for {target_date_str}")
        return None


def calculate_daily_averages(hourly_data):
    """Calculate daily averages from hourly API data"""
    if not hourly_data:
        return None

    avg_data = {}
    numeric_fields = ['temp', 'dew', 'humidity', 'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
                      'precip']

    for field in numeric_fields:
        values = [item[field] for item in hourly_data if field in item]
        if values:
            avg_data[field] = sum(values) / len(values)
        else:
            avg_data[field] = 0

    # Use the most common condition
    conditions = [item.get('condition', 'Clear') for item in hourly_data]
    avg_data['condition'] = max(set(conditions), key=conditions.count)

    return avg_data


def create_prediction_df(data, target_date):
    """Create prediction DataFrame from API data"""
    pred_dict = {'datetime': target_date}

    # Map all required fields
    pred_dict['temp'] = round(data.get('temp', 0), 2)
    pred_dict['dew'] = round(data.get('dew', 0), 2)
    pred_dict['humidity'] = round(data.get('humidity', 0), 2)
    pred_dict['windspeed'] = round(data.get('windspeed', 0), 2)
    pred_dict['winddir'] = round(data.get('winddir', 0), 2)
    pred_dict['sealevelpressure'] = round(data.get('sealevelpressure', 0), 2)
    pred_dict['cloudcover'] = round(data.get('cloudcover', 0), 2)
    pred_dict['visibility'] = round(data.get('visibility', 0), 2)
    pred_dict['precip'] = round(data.get('precip', 0), 2)

    return pd.DataFrame([pred_dict])


def predict_for_date(target_date, use_api=True):
    """Predict weather for a specific date with smart fallback"""
    global model, scalers, feature_cols, df, last_used_api

    # Initialize API cache on first prediction
    if not api_initialized and use_api:
        initialize_api_cache()

    # Parse date
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date, format='%Y-%m-%d', errors='coerce')

    if pd.isna(target_date):
        raise ValueError("❌ Invalid date format! Use 'YYYY-MM-DD'")

    target_date_str = target_date.strftime('%Y-%m-%d')

    # ✅ PRIORITY 1 & 2: Try API first (cache + fresh API)
    if use_api:
        api_prediction = get_api_prediction(target_date)
        if api_prediction is not None:
            source = "cached API" if last_used_api else "fresh API"
            print(f"🎯 RESULT SOURCE: {source} data for {target_date_str}")
            return api_prediction

    # ✅ PRIORITY 3: Fall back to model prediction (API failed)
    print(f"🔄 API unavailable, using model prediction for {target_date_str}")
    last_used_api = False

    if model is None:
        initialize_model()

    last_date = df['datetime'].iloc[-1]
    days_ahead = int((target_date - last_date).days)

    print(f"📅 Model predicting for {target_date.date()} ({days_ahead} days ahead)...")

    # Create input sequence from last lookback days
    input_sequence = []
    for i in range(lookback):
        row_idx = len(df) - lookback + i
        row_data = []
        for col in feature_cols:
            value = df[col].iloc[row_idx]
            scaled_value = scalers[col].transform([[value]])[0, 0]
            row_data.append(scaled_value)
        input_sequence.append(row_data)

    current_sequence = np.array(input_sequence)

    # Predict iteratively
    for day in range(days_ahead):
        input_seq = np.expand_dims(current_sequence, axis=0)
        pred_scaled = model.predict(input_seq, verbose=0)
        current_sequence = np.vstack([current_sequence[1:], pred_scaled])

    # Rescale prediction
    pred_rescaled = {}
    for i, col in enumerate(feature_cols):
        pred_rescaled[col] = float(scalers[col].inverse_transform([[pred_scaled[0, i]]])[0, 0])

    # Create result
    pred_dict = {'datetime': target_date}
    pred_dict.update(pred_rescaled)
    result_df = pd.DataFrame([pred_dict])

    print(f"🎯 RESULT SOURCE: AI Model prediction for {target_date_str}")
    print("✅ Model prediction completed!")
    return result_df


def determine_octant(lat, lon):
    """Determine octant of the globe (STANAG 4082)"""
    if lat >= 0 and lon >= 0:
        return "1"  # NE
    elif lat >= 0 and lon < 0:
        return "2"  # SE
    elif lat < 0 and lon >= 0:
        return "3"  # SW
    else:
        return "4"  # NW


def encode_latitude(lat):
    """Encode latitude as tens, units and tenths of degrees"""
    lat_abs = abs(lat)
    tens = int(lat_abs // 10)
    units = int(lat_abs % 10)
    tenths = int((lat_abs * 10) % 10)
    return f"{tens:01d}{units:01d}{tenths:01d}"


def encode_longitude(lon, octant):
    """Encode longitude as tens, units and tenths of degrees"""
    lon_abs = abs(lon)
    tens = int(lon_abs // 10)
    units = int(lon_abs % 10)
    tenths = int((lon_abs * 10) % 10)
    return f"{tens:01d}{units:01d}{tenths:01d}"


def encode_wind_direction(wind_dir_degrees):
    """Convert wind direction from degrees to mils (001-640)"""
    # 1 degree = 17.777 mils (approximately)
    wind_dir_mils = int(round(wind_dir_degrees * 17.777777))
    wind_dir_mils = max(1, min(640, wind_dir_mils))  # Clamp to 001-640
    return f"{wind_dir_mils:03d}"


def encode_wind_speed(wind_speed_kmh):
    """Convert wind speed from km/h to knots and encode"""
    wind_speed_knots = int(round(wind_speed_kmh * 0.539957))
    wind_speed_knots = max(0, min(999, wind_speed_knots))  # Clamp to reasonable range
    return f"{wind_speed_knots:03d}"


def encode_temperature(temp_kelvin):
    """Encode temperature in 0.1 Kelvin units"""
    temp_encoded = int(round(temp_kelvin * 10))
    temp_encoded = max(0, min(9999, temp_encoded))  # Clamp to reasonable range
    return f"{temp_encoded:04d}"


def encode_pressure(pressure_hpa):
    """Encode pressure in millibars"""
    pressure_encoded = int(round(pressure_hpa))
    pressure_encoded = max(0, min(9999, pressure_encoded))  # Clamp to reasonable range
    return f"{pressure_encoded:04d}"


def generate_metcm_message(weather_data, lat=18.5204, lon=73.8567, altitude=560):
    """
    Generate METCM message according to STANAG 4082 artillery meteorological format
    """
    try:
        from datetime import datetime, timezone

        # Extract weather parameters
        temp = weather_data.get('temp', 0)
        wind_speed = weather_data.get('windspeed', 0)
        wind_dir = weather_data.get('winddir', 0)
        pressure = weather_data.get('sealevelpressure', 1013)

        # Convert temperature from Celsius to Kelvin
        temp_k = temp + 273.15

        # Get current time in UTC
        current_time = datetime.now(timezone.utc)

        # Q - Octant of the Globe (1=NE, 2=SE, 3=SW, 4=NW)
        q = determine_octant(lat, lon)

        # LaLaLa - Latitude in tens, units and tenths of degrees
        lat_encoded = encode_latitude(lat)

        # LoLoLo - Longitude in tens, units and tenths of degrees
        lon_encoded = encode_longitude(lon, q)

        # YY - Day of the month (01 to 30)
        yy = current_time.strftime("%d")

        # GoGoGo - Time of beginning of validity period (GMT in whole hours and tenths)
        gogogo = current_time.strftime("%H") + "0"  # Current hour + 0 tenths

        # G - Duration of validity in hours (1-8; 9=12 hours)
        g = "1"  # Default 1 hour validity

        # HHH - Altitude of Met Datum Plane in tens of meters above MSL
        hhh = f"{int(altitude / 10):03d}"

        # PPP - Pressure at MDP in millibars (omit thousands digit)
        ppp = f"{int(pressure % 1000):03d}"

        # Header line
        header = f"METCM{q}{lat_encoded}{lon_encoded}{yy}{gogogo}{g}{hhh}{ppp}"

        # Generate ALL zones from 00 to 26
        zones = []

        # Zone 00 - Surface observation
        ddd_00 = encode_wind_direction(wind_dir)
        fff_00 = encode_wind_speed(wind_speed)
        ttt_00 = encode_temperature(temp_k)
        rrrr_00 = encode_pressure(pressure)
        zones.append(f"00{ddd_00}{fff_00} {ttt_00}{rrrr_00}")

        # Generate zones 01 to 24 with gradual atmospheric changes
        for zone in range(1, 25):
            # Calculate atmospheric changes with altitude
            altitude_factor = zone / 24.0

            # Wind direction changes with altitude (veering in Northern Hemisphere)
            wind_dir_zone = (wind_dir + (zone * 15)) % 360

            # Wind speed increases with altitude
            wind_speed_zone = wind_speed + (zone * 2)

            # Temperature decreases with altitude (~6.5°C per km)
            temp_k_zone = temp_k - (zone * 2.5)

            # Pressure decreases with altitude
            pressure_zone = pressure - (zone * 25)

            ddd_zone = encode_wind_direction(wind_dir_zone)
            fff_zone = encode_wind_speed(wind_speed_zone)
            ttt_zone = encode_temperature(temp_k_zone)
            rrrr_zone = encode_pressure(pressure_zone)

            zones.append(f"{zone:02d}{ddd_zone}{fff_zone} {ttt_zone}{rrrr_zone}")

        # Zone 25 - Upper atmospheric zone
        ddd_25 = encode_wind_direction((wind_dir + 375) % 360)
        fff_25 = encode_wind_speed(wind_speed + 50)
        ttt_25 = encode_temperature(temp_k - 62.5)
        rrrr_25 = encode_pressure(pressure - 625)
        zones.append(f"25{ddd_25}{fff_25} {ttt_25}{rrrr_25}")

        # Zone 26 - Top atmospheric zone
        ddd_26 = encode_wind_direction((wind_dir + 390) % 360)
        fff_26 = encode_wind_speed(wind_speed + 52)
        ttt_26 = encode_temperature(temp_k - 65)
        rrrr_26 = encode_pressure(pressure - 650)
        zones.append(f"26{ddd_26}{fff_26} {ttt_26}{rrrr_26}")

        # Message terminator
        terminator = "99999"

        # Combine all parts
        metcm_lines = [header] + zones + [terminator]
        metcm_message = "\n".join(metcm_lines)

        return metcm_message

    except Exception as e:
        print(f"❌ METCM generation failed: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return "METCM GENERATION ERROR"


def generate_hourly_metcm(hourly_data):
    """Generate METCM messages for each hourly data point"""
    hourly_metcm = []

    for hour_data in hourly_data:
        try:
            metcm_message = generate_metcm_message(hour_data)
            metcm_explanation = get_metcm_explanation(metcm_message)

            hourly_metcm.append({
                'hour': hour_data['hour'],
                'metcm': metcm_message,
                'explanation': metcm_explanation
            })
        except Exception as e:
            print(f"❌ Hourly METCM generation failed for {hour_data.get('hour', 'unknown')}: {e}")
            hourly_metcm.append({
                'hour': hour_data['hour'],
                'metcm': 'METCM GENERATION ERROR',
                'explanation': 'Failed to generate METCM data for this hour'
            })

    return hourly_metcm


def get_metcm_explanation(metcm_message):
    """Provide explanation of METCM message format"""
    explanation = """METCM Message Explanation (STANAG 4082):

Header Format: METCMQ LaLaLaLoLoLo YYGoGoGoGHHHPPP

• Q: Octant of the Globe (1=NE, 2=SE, 3=SW, 4=NW)
• LaLaLa: Latitude in tens, units and tenths of degrees
• LoLoLo: Longitude in tens, units and tenths of degrees  
• YY: Day of the month (01-30)
• GoGoGo: Time of validity period start (GMT hours + tenths)
• G: Duration of validity in hours (1-8; 9=12 hours)
• HHH: Altitude of Met Datum Plane (tens of meters MSL)
• PPP: Pressure at MDP (millibars, omit thousands)

Zone Data: ZZDDDFFF TTTRRRR

• ZZ: Zone number (00=surface, 01-26=atmospheric zones)
• DDD: Mean wind direction in tens of mils (001-640)
• FFF: Mean wind speed in knots
• TTTT: Mean virtual temperature in 0.1 Kelvin
• RRRR: Zone midpoint pressure in millibars

99999: Message terminator

This message provides artillery meteorological data for ballistic calculations.
"""
    return explanation


# Test function
if __name__ == "__main__":
    try:
        # Test initialization
        print("🧪 Testing smart caching system...")
        initialize_api_cache()

        # Test prediction for tomorrow
        tomorrow = datetime.now() + timedelta(days=1)
        result = predict_for_date(tomorrow)
        print("\n📊 Prediction Result:")
        print(result)

        # Show cache info
        cache_data = load_api_cache()
        if cache_data:
            print(f"\n💾 Cache info:")
            print(f"Last updated: {cache_data['last_updated']}")
            print(f"Days in cache: {len(cache_data['forecasts'])}")
            for date in sorted(cache_data['forecasts'].keys())[:5]:  # Show first 5 dates
                print(f"  - {date}: {len(cache_data['forecasts'][date])} forecasts")

    except Exception as e:
        print(f"❌ Test failed: {e}")
