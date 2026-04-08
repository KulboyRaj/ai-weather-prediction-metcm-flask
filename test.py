import requests
import json
from datetime import datetime, timedelta


def test_weather_api():
    """Test what dates the OpenWeatherMap API actually returns"""

    API_KEY = "999cb96a61f49748b8a5c51c317dce55"
    lat, lon = 18.5204, 73.8567  # Pune coordinates

    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    print("🧪 Testing OpenWeatherMap API...")
    print(f"🔗 API URL: {url}")
    print(f"📍 Location: Pune ({lat}, {lon})")
    print(f"🔑 API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    print("-" * 50)

    try:
        response = requests.get(url, timeout=10)
        print(f"📡 Response Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            # Print API response summary
            print(f"📊 Total forecast points: {len(data['list'])}")
            print(f"📍 City: {data['city']['name']}, {data['city']['country']}")
            print("-" * 50)

            # Extract and display all dates from API response
            forecasts = {}
            for item in data['list']:
                dt = datetime.fromtimestamp(item['dt'])
                date_str = dt.strftime('%Y-%m-%d')
                time_str = dt.strftime('%H:%M')

                if date_str not in forecasts:
                    forecasts[date_str] = []

                forecasts[date_str].append({
                    'time': time_str,
                    'temp': item['main']['temp'],
                    'condition': item['weather'][0]['main']
                })

            # Display dates and data points
            print("📅 DATES RETURNED BY API:")
            print("-" * 50)
            for date in sorted(forecasts.keys()):
                print(f"📅 {date}: {len(forecasts[date])} data points")
                for forecast in forecasts[date][:2]:  # Show first 2 time slots
                    print(f"   ⏰ {forecast['time']}: {forecast['temp']}°C, {forecast['condition']}")
                if len(forecasts[date]) > 2:
                    print(f"   ... and {len(forecasts[date]) - 2} more time slots")
                print()

            # Check specific dates
            today = datetime.now().date()
            print("🔍 DATE ANALYSIS:")
            print("-" * 50)
            for i in range(1, 8):  # Check next 7 days
                check_date = today + timedelta(days=i)
                check_date_str = check_date.strftime('%Y-%m-%d')
                exists = check_date_str in forecasts
                status = "✅ AVAILABLE" if exists else "❌ NOT AVAILABLE"
                print(f"{status} - {check_date_str} (Day +{i})")

            return forecasts

        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return None

    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        return None


def test_specific_date(target_date_str):
    """Test if a specific date is available in API response"""
    forecasts = test_weather_api()

    if forecasts:
        print("\n" + "=" * 60)
        print(f"🎯 SPECIFIC DATE TEST: {target_date_str}")
        print("=" * 60)

        if target_date_str in forecasts:
            print(f"✅ SUCCESS: {target_date_str} is available in API response!")
            print(f"📊 Data points: {len(forecasts[target_date_str])}")
            print("Sample data:")
            for i, point in enumerate(forecasts[target_date_str][:3]):
                print(f"  {i + 1}. {point['time']} - {point['temp']}°C, {point['condition']}")
        else:
            print(f"❌ FAIL: {target_date_str} is NOT available in API response")
            available_dates = sorted(forecasts.keys())
            print(f"📅 Available dates: {', '.join(available_dates)}")


if __name__ == "__main__":
    # Test the API
    print("🚀 OPENWEATHERMAP API TEST")
    print("=" * 60)

    # Test general API functionality
    forecasts = test_weather_api()

    # Test specific date (November 1st, 2025)
    test_specific_date("2025-11-01")

    # Also test tomorrow's date for comparison
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    test_specific_date(tomorrow)