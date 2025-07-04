import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import random

LOOK_BACK = 14
EPOCHS = 100
BATCH_SIZE = 16
LR = 0.001
DEVICE = "cpu"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

os.makedirs("src/results", exist_ok=True)

def forecast_accuracy(forecast, actual):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def create_dataset(series, look_back):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back]) # at each step the forecasted value becomes the next value in the series, removing the first value in the window
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

df = pd.read_csv("src/data/cleaned_sleep_timeseries.csv")
df["date"] = pd.to_datetime(df["index"])
df.set_index("date", inplace=True)
series = df["sleep_duration"].values

scaler = StandardScaler()
scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten() # scale the series to have mean=0 and std=1

X, y = create_dataset(scaled_series, LOOK_BACK)

split_idx = int(len(X) * 0.8) # 80% for training, 20% for testing
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16) # first layer with 16 neurons
        self.relu = nn.ReLU() # ReLU activation function
        self.dropout = nn.Dropout(0.2) # dropout layer with 20% dropout rate
        self.fc2 = nn.Linear(16, 1) # output layer with 1 neuron (for regression)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = MLP(input_dim=LOOK_BACK).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")

# inference on test set
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    preds_scaled = model(X_test_tensor).squeeze().cpu().numpy()

preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# evaluation on test set
metrics = forecast_accuracy(preds, y_true)
print("\nForecast Accuracy:")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")

start_date = df.index[LOOK_BACK + split_idx]
dates = pd.date_range(start=start_date, periods=len(preds))
forecast_df = pd.DataFrame({
    "date": dates,
    "model": "MLP",
    "prediction": preds
})
forecast_df.to_csv("src/results/mlp_forecast.csv", index=False)

# forecast vs actual plot
plt.figure(figsize=(10, 5))
plt.plot(dates, y_true, label="Actual")
plt.plot(dates, preds, label="MLP Forecast", linestyle="--")
plt.title("MLP Forecast vs Actual Sleep Duration")
plt.xlabel("Date")
plt.ylabel("Sleep Duration (minutes)")
plt.legend()
plt.tight_layout()
plt.show()
