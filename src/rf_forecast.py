import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

LOOK_BACK = 12
N_ESTIMATORS = 100
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

os.makedirs("src/results", exist_ok=True)

def forecast_accuracy(forecast, actual):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

df = pd.read_csv("src/data/cleaned_sleep_timeseries.csv")
df["date"] = pd.to_datetime(df["index"])
df.set_index("date", inplace=True)
series = df["sleep_duration"].values

def create_lag_features(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i]) # window creation where the last predicted value becomes the next value in the series
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_lag_features(series, LOOK_BACK)

split_idx = int(len(X) * 0.8) # 80% for training, 20% for testing
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# recursive forecasting, where the last predicted value becomes the next value in the series
window = series[split_idx:split_idx + LOOK_BACK].copy()
rf_forecast = []

for _ in range(len(y_test)):
    pred = model.predict(window.reshape(1, -1))[0]
    rf_forecast.append(pred)
    window = np.roll(window, -1)
    window[-1] = pred  # update last value with prediction

# evaluation on test set
metrics = forecast_accuracy(rf_forecast, y_test)
print("\nForecast Accuracy:")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

dates = df.index[split_idx + LOOK_BACK:]
forecast_df = pd.DataFrame({
    "date": dates[:len(rf_forecast)],
    "model": "RandomForest",
    "prediction": rf_forecast
})
forecast_df.to_csv("src/results/rf_forecast.csv", index=False)

# forecast data vs actual plot
plt.figure(figsize=(10, 5))
plt.plot(dates, y_test, label="Actual")
plt.plot(dates, rf_forecast, label="RF Forecast", linestyle="--")
plt.title("Random Forest Forecast vs Actual Sleep Duration")
plt.xlabel("Date")
plt.ylabel("Sleep Duration (minutes)")
plt.legend()
plt.tight_layout()
plt.show()
