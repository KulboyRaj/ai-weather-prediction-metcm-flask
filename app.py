from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import model_predictor as predictor
import os
import numpy as np

app = Flask(__name__)

# Initialize API cache when app starts
print("🚀 Starting Weather Forecasting App...")
print("🔄 Initializing API cache...")
predictor.initialize_api_cache()
print("✅ App initialization complete!")


@app.route('/')
def index():
    # Get tomorrow's date by default
    tomorrow = datetime.now() + timedelta(days=1)
    default_date = tomorrow.strftime('%Y-%m-%d')

    # Calculate min and max dates
    min_date = tomorrow
    max_date = tomorrow + timedelta(days=365)

    return render_template('index.html',
                           default_date=default_date,
                           min_date=min_date.strftime('%Y-%m-%d'),
                           max_date=max_date.strftime('%Y-%m-%d'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        target_date = data.get('date')

        # Your existing prediction code...
        prediction_result = predictor.predict_for_date(target_date)

        # Check if we used API (cache) or model
        used_api = getattr(predictor, 'last_used_api', False)

        # Get condition from prediction result if available, otherwise calculate
        if 'condition' in prediction_result.columns:
            actual_condition = prediction_result['condition'].iloc[0]
        else:
            # Calculate condition based on weather parameters for model predictions
            actual_condition = calculate_condition_from_data(prediction_result.iloc[0])

        # Generate summary data
        summary = {
            'max_temp': round(prediction_result['temp'].max(), 1),
            'min_temp': round(prediction_result['temp'].min(), 1),
            'avg_humidity': round(prediction_result['humidity'].mean(), 1),
            'max_wind': round(prediction_result['windspeed'].max(), 1),
            'total_precip': round(prediction_result['precip'].sum(), 1),
            'dominant_condition': actual_condition  # Use actual or calculated condition
        }

        # Generate hourly data (2-hour intervals)
        hourly_data = []
        times = ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00',
                 '12:00', '14:00', '16:00', '18:00', '20:00', '22:00']

        for i, time in enumerate(times):
            # Calculate base values
            base_temp = prediction_result['temp'].iloc[0]
            base_humidity = prediction_result['humidity'].iloc[0]
            base_cloudcover = prediction_result['cloudcover'].iloc[0]
            base_precip = prediction_result['precip'].iloc[0]

            # Calculate hourly variations
            temp_variation = base_temp + (i - 6) * 2
            humidity_variation = base_humidity * (0.9 + (i % 4) * 0.05)
            cloudcover_variation = base_cloudcover * (0.8 + (i % 2) * 0.2)
            precip_variation = base_precip * (0.5 + (i % 2) * 0.5)

            # Determine condition for this hour
            if used_api and 'condition' in prediction_result.columns:
                # Use actual condition from API for all hours
                hour_condition = actual_condition
            else:
                # Calculate condition based on hourly parameters for model
                hour_condition = calculate_condition_from_params(
                    temp=temp_variation,
                    humidity=humidity_variation,
                    cloudcover=cloudcover_variation,
                    precip=precip_variation
                )

            hourly_data.append({
                'hour': time,
                'temp': round(temp_variation, 1),
                'dew': round(prediction_result['dew'].iloc[0] + (i - 6) * 1, 1),
                'humidity': round(humidity_variation, 1),
                'windspeed': round(prediction_result['windspeed'].iloc[0] * (0.8 + (i % 3) * 0.2), 1),
                'winddir': (prediction_result['winddir'].iloc[0] + i * 5) % 360,
                'sealevelpressure': round(prediction_result['sealevelpressure'].iloc[0] + (i - 6) * 0.5, 1),
                'cloudcover': round(cloudcover_variation, 1),
                'visibility': round(prediction_result['visibility'].iloc[0] * (0.9 + (i % 3) * 0.1), 1),
                'precip': round(precip_variation, 1),
                'condition': hour_condition
            })

        # Generate METCM for each hourly data point
        hourly_metcm = predictor.generate_hourly_metcm(hourly_data)

        # Generate daily METCM
        daily_metcm = predictor.generate_metcm_message(hourly_data[6])
        daily_metcm_explanation = predictor.get_metcm_explanation(daily_metcm)

        return jsonify({
            'success': True,
            'date': target_date,
            'summary': summary,
            'hourly_data': hourly_data,
            'hourly_metcm': hourly_metcm,
            'metcm': daily_metcm,
            'metcm_explanation': daily_metcm_explanation,
            'data_source': 'API Cache' if used_api else 'AI Model'
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        print(f"Detailed traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)})


def calculate_condition_from_data(data_row):
    """Calculate weather condition from data row for model predictions"""
    return calculate_condition_from_params(
        temp=data_row['temp'],
        humidity=data_row['humidity'],
        cloudcover=data_row['cloudcover'],
        precip=data_row['precip']
    )


def calculate_condition_from_params(temp, humidity, cloudcover, precip):
    """Calculate weather condition based on weather parameters"""
    if precip > 0.5:
        if temp < 2:
            return 'Snow'
        else:
            return 'Rain'
    elif cloudcover > 80:
        return 'Cloudy'
    elif cloudcover > 50:
        return 'Partly Cloudy'
    elif cloudcover > 20:
        return 'Mostly Clear'
    else:
        return 'Clear'
def convert_to_timeslots(daily_prediction_df):
    """Convert daily prediction to 12 time slots (2-hour gaps) with realistic variations"""
    timeslots = []

    # Extract the daily prediction row
    daily_data = daily_prediction_df.iloc[0]

    # Base values
    base_temp = daily_data['temp']
    base_dew = daily_data['dew']
    base_humidity = daily_data['humidity']
    base_windspeed = daily_data['windspeed']
    base_winddir = daily_data['winddir']
    base_pressure = daily_data['sealevelpressure']
    base_cloudcover = daily_data['cloudcover']
    base_visibility = daily_data['visibility']
    base_precip = daily_data['precip']

    # Time slots: 00:00, 02:00, 04:00, ..., 22:00 (12 slots)
    for slot in range(12):
        hour = slot * 2
        time_str = f"{hour:02d}:00"

        # Calculate realistic variations while maintaining daily average

        # Temperature: follows diurnal cycle (coolest at 4 AM, warmest at 2 PM)
        temp_variation = calculate_temperature_variation(hour, base_temp)
        final_temp = base_temp + temp_variation

        # Dew point: follows similar pattern but less variation
        dew_variation = temp_variation * 0.3  # Dew point varies less than temperature
        final_dew = base_dew + dew_variation

        # Humidity: inverse relationship with temperature
        humidity_variation = calculate_humidity_variation(hour, base_humidity)
        final_humidity = max(20, min(100, base_humidity + humidity_variation))

        # Wind speed: often higher during daytime
        wind_variation = calculate_wind_variation(hour, base_windspeed)
        final_windspeed = max(0, base_windspeed + wind_variation)

        # Wind direction: slight variations
        final_winddir = (base_winddir + (hour * 2)) % 360

        # Pressure: very slight diurnal variation
        pressure_variation = np.sin(2 * np.pi * (hour - 10) / 24) * 0.3
        final_pressure = base_pressure + pressure_variation

        # Cloud cover: often follows specific patterns
        cloud_variation = calculate_cloud_variation(hour, base_cloudcover)
        final_cloudcover = max(0, min(100, base_cloudcover + cloud_variation))

        # Visibility: often better during day
        visibility_variation = calculate_visibility_variation(hour, base_visibility)
        final_visibility = max(0, base_visibility + visibility_variation)

        # Precipitation: higher probability during certain hours
        precip_variation = calculate_precip_variation(hour, base_precip)
        final_precip = max(0, base_precip + precip_variation)

        timeslots.append({
            'hour': time_str,
            'temp': round(final_temp, 1),
            'dew': round(final_dew, 1),
            'humidity': round(final_humidity, 1),
            'windspeed': round(final_windspeed, 1),
            'winddir': round(final_winddir),
            'sealevelpressure': round(final_pressure, 1),
            'cloudcover': round(final_cloudcover, 1),
            'visibility': round(final_visibility, 1),
            'precip': round(final_precip, 2),
            'condition': get_weather_condition(final_temp, final_humidity, final_cloudcover, final_precip)
        })

    return timeslots


def calculate_temperature_variation(hour, base_temp):
    """Calculate realistic temperature variation throughout the day"""
    # Temperature follows a sine wave: coolest at 4 AM, warmest at 2 PM
    amplitude = 6  # ±6°C variation
    phase_shift = 4  # Coolest at 4 AM

    variation = -amplitude * np.cos(2 * np.pi * (hour - phase_shift) / 24)
    return variation


def calculate_humidity_variation(hour, base_humidity):
    """Calculate realistic humidity variation (inverse of temperature)"""
    # Humidity is higher at night, lower during day
    amplitude = 15  # ±15% variation
    phase_shift = 16  # Highest humidity around 4 PM (before evening)

    variation = amplitude * np.sin(2 * np.pi * (hour - phase_shift) / 24)
    return variation


def calculate_wind_variation(hour, base_windspeed):
    """Calculate realistic wind speed variation"""
    # Wind often picks up during daytime
    amplitude = min(2, base_windspeed * 0.3)  # Limit variation
    phase_shift = 14  # Windiest around 2 PM

    variation = amplitude * np.sin(2 * np.pi * (hour - phase_shift) / 24)
    return variation


def calculate_cloud_variation(hour, base_cloudcover):
    """Calculate realistic cloud cover variation"""
    # Clouds often build up during day, clear at night
    amplitude = min(20, base_cloudcover * 0.4)  # Limit variation
    phase_shift = 16  # Most cloudy late afternoon

    variation = amplitude * np.sin(2 * np.pi * (hour - phase_shift) / 24)
    return variation


def calculate_visibility_variation(hour, base_visibility):
    """Calculate realistic visibility variation"""
    # Better visibility during day
    amplitude = min(2, base_visibility * 0.2)
    variation = amplitude * np.sin(2 * np.pi * (hour - 12) / 24)
    return variation


def calculate_precip_variation(hour, base_precip):
    """Calculate realistic precipitation variation"""
    # Higher precipitation probability in afternoon/evening
    if base_precip > 0:
        amplitude = base_precip * 0.5
        phase_shift = 16  # Highest precipitation probability late afternoon
        variation = amplitude * np.sin(2 * np.pi * (hour - phase_shift) / 24)
        return variation
    return 0


def get_weather_condition(temp, humidity, cloudcover, precip):
    """Determine weather condition based on multiple factors"""
    if precip > 0.5:
        if temp < 2:
            return 'Snow'
        else:
            return 'Rain'
    elif cloudcover > 80:
        return 'Cloudy'
    elif cloudcover > 50:
        return 'Partly Cloudy'
    elif cloudcover > 20:
        return 'Mostly Clear'
    else:
        return 'Clear'


def get_weather_summary(daily_prediction):
    """Generate weather summary from daily prediction"""
    daily_data = daily_prediction.iloc[0]

    return {
        'max_temp': round(daily_data['temp'] + 3, 1),  # Estimate max temp
        'min_temp': round(daily_data['temp'] - 3, 1),  # Estimate min temp
        'avg_temp': round(daily_data['temp'], 1),
        'avg_humidity': round(daily_data['humidity'], 1),
        'max_wind': round(daily_data['windspeed'] + 2, 1),  # Estimate max wind
        'total_precip': round(daily_data['precip'], 2),
        'avg_cloudcover': round(daily_data['cloudcover'], 1),
        'dominant_condition': get_weather_condition(
            daily_data['temp'],
            daily_data['humidity'],
            daily_data['cloudcover'],
            daily_data['precip']
        )
    }


if __name__ == '__main__':
    # Check if model and data files exist
    if not os.path.exists('best_weather_lstm.h5'):
        print("❌ Model file 'best_weather_lstm.h5' not found!")
        exit(1)

    if not os.path.exists('weather_cleaned.csv'):
        print("❌ Data file 'weather_cleaned.csv' not found!")
        exit(1)

    print("📧 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0')
