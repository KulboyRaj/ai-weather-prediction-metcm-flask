import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 1. Load and prepare dataset
# -------------------------------
df = pd.read_csv('weather_cleaned.csv')

# Parse date safely and consistently
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

feature_cols = [col for col in df.columns if col != 'datetime']

# Scale features individually
scalers = {}
scaled_data = np.zeros((len(df), len(feature_cols)))
for i, col in enumerate(feature_cols):
    scalers[col] = MinMaxScaler()
    scaled_data[:, i] = scalers[col].fit_transform(df[[col]]).flatten()

lookback = 30

# -------------------------------
# 2. Load trained model
# -------------------------------
model = load_model('best_weather_lstm.h5', compile=False)
print("✅ Model loaded successfully!")

# -------------------------------
# 3. Predict for a specific date
# -------------------------------
def predict_for_date(target_date):
    # Ensure target date is parsed properly
    target_date = pd.to_datetime(target_date, dayfirst=True, errors='coerce')
    if pd.isna(target_date):
        raise ValueError("❌ Invalid date format! Please use 'DD/MM/YYYY'. Example: '06/10/2025'")

    # Get the last available date in dataset
    last_date = df['datetime'].iloc[-1]

    # Compute how many days ahead to forecast
    days_ahead = int((target_date - last_date).days)
    if days_ahead <= 0:
        raise ValueError(f"❌ The date {target_date.date()} must be after the last dataset date ({last_date.date()}).")

    print(f"\n📅 Predicting conditions for {target_date.date()} ({days_ahead} days ahead)...")

    # Start from the last known sequence
    last_sequence = scaled_data[-lookback:]

    # Iteratively predict each future day until the target date
    for _ in range(days_ahead):
        input_seq = np.expand_dims(last_sequence, axis=0)
        pred_scaled = model.predict(input_seq, verbose=0)
        last_sequence = np.vstack([last_sequence[1:], pred_scaled])

    # Rescale the final prediction
    pred_rescaled = [scalers[col].inverse_transform([[pred_scaled[0, i]]])[0, 0]
                     for i, col in enumerate(feature_cols)]

    # Create output dataframe
    pred_dict = {'Predicted_Date': target_date}
    pred_dict.update({col: round(pred_rescaled[i], 2) for i, col in enumerate(feature_cols)})
    result_df = pd.DataFrame([pred_dict])

    print("\n✅ Predicted weather for selected date:\n")
    print(result_df)
    return result_df

# -------------------------------
# 4. Example usage
# -------------------------------
future_prediction = predict_for_date("10/1/2026")  # Change date as needed
