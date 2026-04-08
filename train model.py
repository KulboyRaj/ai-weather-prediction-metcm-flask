import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# --- Load CSV data ---
df = pd.read_csv("weather_cleaned.csv")

# --- Parse datetime ---
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# --- Features to predict ---
feature_cols = ['temp', 'dew', 'humidity', 'windspeed', 'winddir',
                'sealevelpressure', 'cloudcover', 'visibility', 'precip']

# --- Scale each feature individually ---
scalers = {}
scaled_data = np.zeros(df[feature_cols].shape)
for i, col in enumerate(feature_cols):
    scalers[col] = MinMaxScaler()
    scaled_data[:, i] = scalers[col].fit_transform(df[[col]]).flatten()

# --- Create sequences ---
def create_sequences(data, lookback=30):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

lookback = 30
X, y = create_sequences(scaled_data, lookback)

# --- TimeSeries cross-validation ---
tscv = TimeSeriesSplit(n_splits=5)
best_model = None
best_val_loss = np.inf

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\n🧩 Fold {fold+1}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # --- Build LSTM model ---
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, len(feature_cols))),
        Dropout(0.1),
        LSTM(64, return_sequences=False),
        Dropout(0.1),
        Dense(len(feature_cols), activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')

    # --- Early stopping ---
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # --- Train ---
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )

    val_loss = min(history.history['val_loss'])
    print(f"Fold {fold+1} validation loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

# --- Save trained model ---
best_model.save("best_weather_lstm.h5")
print("\n✅ Best LSTM model saved as 'best_weather_lstm.h5'")

# --- Function to predict future days ---
def forecast_future_days(start_date=None, days_ahead=5):
    if start_date:
        start_date = pd.to_datetime(start_date, dayfirst=True, errors='coerce')
        idx = df.index[df['datetime'] == start_date].tolist()
        if not idx:
            raise ValueError("Start date not found in dataset!")
        idx = idx[0]
    else:
        idx = len(df) - 1  # use last available date

    predictions = []
    last_sequence = scaled_data[idx - lookback + 1 : idx + 1]

    for day in range(days_ahead):
        input_seq = np.expand_dims(last_sequence, axis=0)
        pred_scaled = best_model.predict(input_seq, verbose=0)
        pred_rescaled = np.array([scalers[col].inverse_transform([[pred_scaled[0, i]]])[0,0]
                                  for i, col in enumerate(feature_cols)])

        next_date = df.loc[idx, 'datetime'] + pd.Timedelta(days=day + 1)
        pred_dict = {'Predicted_Date': next_date}
        pred_dict.update({col: pred_rescaled[i] for i, col in enumerate(feature_cols)})
        predictions.append(pred_dict)

        # Append prediction to last_sequence for next day
        last_scaled = np.array(pred_scaled)
        last_sequence = np.vstack([last_sequence[1:], last_scaled])

    pred_df = pd.DataFrame(predictions)
    print(f"\n✅ Predicted {days_ahead} future days:")
    print(pred_df.round(2))
    return pred_df

# --- Example usage ---
future_predictions = forecast_future_days(days_ahead=5)