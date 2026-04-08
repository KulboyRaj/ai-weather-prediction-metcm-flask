import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

# =========================================================
# CONFIGURATION
# =========================================================
MODEL_PATH = "new.h5"
CSV_PATH = "weather_cleaned.csv"
LOOKBACK = 30
EXCEL_OUTPUT = "lstm_weather_model_metrics.xlsx"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(CSV_PATH)

df['datetime'] = pd.to_datetime(
    df['datetime'],
    dayfirst=True,
    errors='coerce'
)

df = df.dropna(subset=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

if df.empty:
    raise ValueError("Dataset is empty after datetime parsing.")

feature_cols = [
    'temp', 'dew', 'humidity', 'windspeed', 'winddir',
    'sealevelpressure', 'cloudcover', 'visibility', 'precip'
]

# =========================================================
# SCALE FEATURES
# =========================================================
scaled_data = np.zeros((len(df), len(feature_cols)))
scalers = {}

for i, col in enumerate(feature_cols):
    scaler = MinMaxScaler()
    scaled_data[:, i] = scaler.fit_transform(df[[col]]).flatten()
    scalers[col] = scaler

# =========================================================
# CREATE SEQUENCES
# =========================================================
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK)

# =========================================================
# LOAD MODEL (evaluation only)
# =========================================================
model = load_model(MODEL_PATH, compile=False)

# =========================================================
# EVALUATION
# =========================================================
y_pred = model.predict(X, verbose=0)

results = []

for i, col in enumerate(feature_cols):
    y_true = y[:, i]
    y_hat = y_pred[:, i]

    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)

    non_zero = y_true != 0
    mape = np.mean(
        np.abs((y_true[non_zero] - y_hat[non_zero]) / y_true[non_zero])
    ) * 100

    print(f"Feature: {col}")
    print(f"   MSE      : {mse:.6f}")
    print(f"   RMSE     : {rmse:.6f}")
    print(f"   MAE      : {mae:.6f}")
    print(f"   MAPE (%) : {mape:.2f}")
    print(f"   R² Score : {r2:.4f}\n")

    results.append({
        "Feature": col,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R2 Score": r2
    })

# =========================================================
# SAVE METRICS TO EXCEL
# =========================================================
results_df = pd.DataFrame(results)
results_df.to_excel(EXCEL_OUTPUT, index=False)

print(f"Metrics saved to {EXCEL_OUTPUT}")


