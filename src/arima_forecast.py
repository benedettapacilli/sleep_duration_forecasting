import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def forecast_accuracy(forecast, actual):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

df = pd.read_csv("src/data/cleaned_sleep_timeseries.csv")
df["date"] = pd.to_datetime(df["index"])
df.set_index("date", inplace=True)

df.index = pd.DatetimeIndex(df.index)
df = df.asfreq("D")

y = df["sleep_duration"]

split_idx = int(len(y) * 0.8) # 80% for training, 20% for testing
train, test = y[:split_idx], y[split_idx:]

# grid Search for ARIMA(p, d=0, q) with lowest AIC, d=0 because the series is already stationary
best_aic = np.inf
best_order = None
best_model = None

print("Searching best ARIMA(p,0,q) model...")
for p in range(0, 4):
    for q in range(0, 4):
        try:
            model = ARIMA(train, order=(p, 0, q))
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = (p, 0, q)
                best_model = model_fit
        except:
            continue

print(f"Best ARIMA order: {best_order} with AIC = {best_aic:.2f}")

n_test = len(test) # forecasting the same number of steps as the test set
forecast = best_model.forecast(steps=n_test)

metrics = forecast_accuracy(forecast, test)
print("\nForecast Accuracy:")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

# actual data vs forecast plot
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="Forecast", linestyle="--")
plt.title(f"ARIMA{best_order} Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Sleep Duration (minutes)")
plt.legend()
plt.tight_layout()
plt.show()

forecast_df = pd.DataFrame({
    "date": test.index,
    "model": "ARIMA",
    "prediction": forecast.values
})
forecast_df.to_csv("src/results/arima_forecast.csv", index=False)