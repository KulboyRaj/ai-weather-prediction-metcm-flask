# 🌤️ AI Weather Prediction & METCM Message Converter

> **An AI-powered weather forecasting web application** that uses an LSTM deep learning model trained on 3 years of Pune city weather data. It intelligently fetches live forecasts from the OpenWeatherMap API, serves from a local JSON cache when offline, and falls back to the trained AI model when no internet is available. Also generates military-grade **METCM (STANAG 4082)** meteorological messages.

---

## 🔍 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Training the AI Model](#-training-the-ai-model)
- [Running the Application](#-running-the-application)
- [METCM Message Format](#-metcm-message-format-stanag-4082)
- [API & Dataset Sources](#-api--dataset-sources)
- [Configuration](#%EF%B8%8F-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🌐 Smart 3-Tier Prediction Engine
| Priority | Source | When Used |
|----------|--------|-----------|
| 1️⃣ First | **API Cache (JSON)** | If the requested date is already cached locally |
| 2️⃣ Second | **Live OpenWeatherMap API** | If internet is available and date not in cache |
| 3️⃣ Third | **AI/LSTM Model (Offline)** | If no internet and no cache available |

### 📡 Live Weather API Integration
- Fetches real-time 5-day forecast from **OpenWeatherMap API**
- Automatically caches results in `weather_cache.json` for offline use
- Cache stays fresh for 6 hours before re-fetching

### 🤖 AI-Powered Offline Prediction
- **LSTM (Long Short-Term Memory)** neural network
- Trained on **3 years of Pune city historical weather data**
- Predicts: Temperature, Humidity, Wind Speed & Direction, Pressure, Cloud Cover, Visibility, Precipitation, Dew Point

### 🪖 METCM Message Generator (STANAG 4082)
- Generates standard military artillery meteorological messages
- 27 atmospheric zones (surface to upper atmosphere)
- Encodes wind direction (mils), wind speed (knots), temperature (Kelvin), and pressure (mbar)

### 🌡️ Detailed 12-Slot Hourly Breakdown
- Weather data broken into 2-hour intervals (00:00 to 22:00)
- Realistic diurnal variations for all weather parameters
- Condition labels: Clear, Mostly Clear, Partly Cloudy, Cloudy, Rain, Snow

---

## 🔄 How It Works

```
User requests weather for a date
            │
            ▼
  Is date in local cache?
     YES ──► Return cached data
     NO  ──► Is internet available?
                  YES ──► Fetch from OpenWeatherMap API
                           └─► Save to cache ──► Return result
                  NO  ──► Run LSTM Model prediction
                           └─► Return AI-generated forecast
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.x, Flask 2.3.3 |
| AI Model | TensorFlow 2.13.0, Keras LSTM |
| Data Processing | Pandas 2.1.1, NumPy 1.24.3 |
| ML Utilities | Scikit-learn 1.3.0 |
| Weather API | OpenWeatherMap (Free Tier) |
| Caching | JSON file-based local cache |
| Frontend | HTML, CSS, JavaScript |

---

## 📁 Project Structure

```
weather_app/
│
├── app.py                    # Flask web server & main routes
├── model_predictor.py        # Core prediction logic (API, cache, LSTM)
├── train model.py            # LSTM model training script
├── predict the weather.py    # Standalone prediction utility
├── accuracy test.py          # Model accuracy evaluation
├── test.py                   # Unit tests
│
├── best_weather_lstm.h5      # ✅ Pre-trained LSTM model (3 years Pune data)
├── new.h5                    # Alternate model checkpoint
├── weather_cleaned.csv       # Cleaned historical weather dataset
├── weather_cache.json        # Auto-generated API cache file
│
├── templates/
│   └── index.html            # Main web UI template
├── static/
│   ├── css/                  # Stylesheets
│   └── js/                   # JavaScript files
│
└── requirements.txt          # Python dependencies
```

---

## ✅ Prerequisites

Make sure you have the following installed:

- **Python 3.8 – 3.11** (TensorFlow 2.13.0 does NOT support Python 3.12+)
- **pip** (Python package manager)
- **Git** (optional, for cloning)

> ⚠️ **Important**: Use Python ≤ 3.11 to avoid TensorFlow compatibility issues.

Check your Python version:
```bash
python --version
```

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/weather-ai-prediction-metcm.git
cd weather-ai-prediction-metcm
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
```
flask==2.3.3
pandas==2.1.1
numpy==1.24.3
tensorflow==2.13.0
scikit-learn==1.3.0
```

> 💡 TensorFlow installation may take a few minutes. Make sure you have at least 1GB of free disk space.

### Step 4: Add Your OpenWeatherMap API Key

Open `model_predictor.py` and replace the API key on **line 112**:

```python
API_KEY = "your_openweathermap_api_key_here"
```

**How to get a free API key:**
1. Sign up at [https://home.openweathermap.org/users/sign_in](https://home.openweathermap.org/users/sign_in)
2. Go to **API Keys** section in your account
3. Copy your key and paste it in `model_predictor.py`

> 🔒 **Security Tip**: Before pushing to GitHub, move your API key to an environment variable or `.env` file and add it to `.gitignore`.

---

## 🧠 Training the AI Model

> ⚠️ **Skip this step** if you are using the pre-trained model (`best_weather_lstm.h5`) already included in the repo.

Only re-train if you want to use new/different data.

### Step 1: Get Historical Weather Data

Download past weather data from:
> 🌐 [Visual Crossing Weather Query Builder](https://www.visualcrossing.com/weather-query-builder/pune/?v=wizard)

- Select **Pune** (or your city)
- Choose a date range (e.g., 3 years)
- Download as **CSV**
- Save as `weather_cleaned.csv` in the project folder

### Step 2: Run the Training Script

```bash
python "train model.py"
```

**What it does:**
- Loads `weather_cleaned.csv`
- Scales all 9 weather features using MinMaxScaler
- Creates 30-day lookback sequences
- Trains an LSTM model using 5-fold TimeSeriesCrossValidation
- Saves the best model as `best_weather_lstm.h5`

**Expected output:**
```
🧩 Fold 1
Epoch 1/200 ...
...
✅ Best LSTM model saved as 'best_weather_lstm.h5'
```

> ⏱️ Training takes **10–60 minutes** depending on your hardware (CPU/GPU).

---

## ▶️ Running the Application

### Step 1: Ensure Required Files Exist

Before running, verify these files are present:
- ✅ `best_weather_lstm.h5` — pre-trained model
- ✅ `weather_cleaned.csv` — historical dataset
- ✅ `model_predictor.py` — prediction engine
- ✅ `templates/index.html` — web UI

### Step 2: Start the Flask Server

```bash
python app.py
```

**Expected output:**
```
🚀 Starting Weather Forecasting App...
🔄 Initializing API cache...
🌐 Trying API call to fetch missing weather data...
✅ API cache updated successfully
✅ App initialization complete!
📧 Open your browser and go to: http://localhost:5000
```

### Step 3: Open the App in Your Browser

```
http://localhost:5000
```

### Step 4: Use the App

1. **Select a date** using the date picker (tomorrow to 1 year ahead)
2. Click **Predict**
3. View:
   - 📊 Daily weather summary (max/min temp, humidity, wind, precipitation)
   - 🕐 12-slot hourly breakdown (2-hour intervals)
   - 🪖 METCM military meteorological message
   - 🔍 METCM explanation (STANAG 4082 format)
4. The **data source badge** shows whether data came from: `API Cache`, `Live API`, or `AI Model`

---

## 🪖 METCM Message Format (STANAG 4082)

The app generates NATO-standard artillery meteorological messages used for ballistic calculations.

### Header Format
```
METCM[Q][LaLaLa][LoLoLo][YY][GoGoGo][G][HHH][PPP]
```

| Field | Description |
|-------|-------------|
| `Q` | Octant of globe (1=NE, 2=SE, 3=SW, 4=NW) |
| `LaLaLa` | Latitude in tens, units, tenths of degrees |
| `LoLoLo` | Longitude in tens, units, tenths of degrees |
| `YY` | Day of month |
| `GoGoGo` | Start of validity period (GMT hours + tenths) |
| `G` | Validity duration in hours |
| `HHH` | Altitude of Met Datum Plane (tens of meters MSL) |
| `PPP` | Pressure at MDP (mbar, omit thousands digit) |

### Zone Data Format
```
ZZ DDD FFF  TTTT RRRR
```

| Field | Description |
|-------|-------------|
| `ZZ` | Zone number (00=surface, 01–26=atmosphere) |
| `DDD` | Mean wind direction in mils (001–640) |
| `FFF` | Mean wind speed in knots |
| `TTTT` | Mean virtual temperature in 0.1 Kelvin |
| `RRRR` | Zone midpoint pressure in millibars |

Message ends with: `99999`

---

## 🌐 API & Dataset Sources

| Source | URL | Purpose |
|--------|-----|---------|
| OpenWeatherMap | [openweathermap.org](https://home.openweathermap.org/users/sign_in) | Live weather API (free tier) |
| Visual Crossing | [visualcrossing.com](https://www.visualcrossing.com/weather-query-builder/pune/?v=wizard) | Historical weather CSV download |

---

## ⚙️ Configuration

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `API_KEY` | `model_predictor.py` | `"your_key"` | OpenWeatherMap API key |
| `lat, lon` | `model_predictor.py` | `18.5204, 73.8567` | City coordinates (Pune) |
| `lookback` | `train model.py` | `30` | LSTM lookback window (days) |
| `max_age_hours` | `model_predictor.py` | `6` | Cache freshness duration |
| `altitude` | `model_predictor.py` | `560` | Altitude for METCM (meters) |

**To switch city**, update coordinates in `model_predictor.py`:
```python
def fetch_weather_api(lat=YOUR_LAT, lon=YOUR_LON):
```
And retrain the model with data for your new city.

---

## 🛠️ Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'tensorflow'`
```bash
pip install tensorflow==2.13.0
```

### ❌ `Model file 'best_weather_lstm.h5' not found!`
Either run `python "train model.py"` to generate it, or ensure the `.h5` file is in the project root.

### ❌ `API request failed with status 401`
Your OpenWeatherMap API key is invalid or not yet activated. New keys can take **up to 2 hours** to activate after signup.

### ❌ TensorFlow not compatible with your Python version
Use Python **3.8 to 3.11**. Check with:
```bash
python --version
```

### ⚠️ Slow first load
The app loads the LSTM model and initializes the cache on startup — this is normal and takes a few seconds.

---

## 📊 Model Architecture

```
Input: [batch, 30 days, 9 features]
     │
     ▼
LSTM(128 units, return_sequences=True)
     │
Dropout(0.1)
     │
LSTM(64 units)
     │
Dropout(0.1)
     │
Dense(9 units, activation='linear')
     │
     ▼
Output: [temp, dew, humidity, windspeed, winddir,
         sealevelpressure, cloudcover, visibility, precip]
```

- **Training**: 5-fold TimeSeriesCV, up to 200 epochs, EarlyStopping (patience=15)
- **Optimizer**: Adam, Loss: MSE
- **Dataset**: 3 years of daily Pune weather data

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 💡 Suggested Repository Name (SEO Optimized)

For maximum GitHub discoverability, consider naming your repo:

```
ai-weather-prediction-metcm-flask
```
or
```
weather-forecast-lstm-metcm-converter
```

**Suggested GitHub Topics/Tags:**
`weather-prediction`, `lstm`, `deep-learning`, `flask`, `metcm`, `stanag-4082`, `openweathermap`, `tensorflow`, `time-series`, `python`, `offline-ai`, `weather-api`

---

<div align="center">
  <p>Made with ❤️ using Python, Flask & TensorFlow</p>
  <p>⭐ Star this repo if you found it helpful!</p>
</div>
