import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import json

def extract_sleep_minutes(row):
    """
    Extract total sleep duration from nested JSON in 'value' field. The Mi Band stores sleep info in a *stringified* JSON format.

    - For 'watch_sleep_report', extract 'night_duration' (total night sleep in minutes).
    - For 'watch_night_sleep_record' and 'watch_daytime_sleep_record', extract 'duration'.
    """
    try:
        record = json.loads(row["value"])
        if row["key"] == "watch_sleep_report":
            return record.get("night_duration")
        elif row["key"] in ["watch_night_sleep_record", "watch_daytime_sleep_record"]:
            return record.get("duration")
        else:
            return None
    except Exception:
        return None

df = pd.read_csv("src/data/user_fitness_data_records.csv")

sleep_keys = [
    "watch_sleep_report",
    "watch_night_sleep_record",
    "watch_daytime_sleep_record"
]
sleep_df = df[df["key"].isin(sleep_keys)].copy()

sleep_df["date"] = pd.to_datetime(sleep_df["time"], unit="s").dt.date # from Unix timestamp to calendar date

sleep_df["sleep_minutes"] = sleep_df.apply(extract_sleep_minutes, axis=1)
sleep_df = sleep_df.dropna(subset=["sleep_minutes"]) # remove rows with missing sleep durations

sns.histplot(sleep_df["sleep_minutes"], kde=True)
plt.title("Raw Sleep Duration Distribution")
plt.xlabel("Sleep duration (minutes)")
plt.show() # sleep duration distribution check: distribution is centered around 400–500 minutes

daily_sleep = sleep_df.groupby("date")["sleep_minutes"].sum().reset_index() # summing night and daytime (naps) sleep durations per day
daily_sleep.columns = ["date", "sleep_duration"]

daily_sleep["date"] = pd.to_datetime(daily_sleep["date"])
daily_sleep.set_index("date", inplace=True)
daily_sleep = daily_sleep.sort_index() # reset index to have a proper time series with dates as the index.

# since forecasting models require a complete time series with no gaps, the series is reindexed to include all days in the range (using NaN for missing days).
all_days = pd.date_range(start=daily_sleep.index.min(), end=daily_sleep.index.max(), freq="D")
daily_sleep = daily_sleep.reindex(all_days)
daily_sleep["sleep_duration"] = daily_sleep["sleep_duration"].interpolate() # missing NAN values are filled with linear interpolation.

result = adfuller(daily_sleep["sleep_duration"].dropna()) # stationarity test using Augmented Dickey-Fuller  (ADF) test
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] <= 0.05:
    print("The series is stationary.")
else:
    print("The series is not stationary")
# result: p-value ≈ 0.012 → The series is already stationary.

daily_sleep["sleep_duration"].plot(title="Daily Sleep Duration")
plt.ylabel("Duration (units from Mi Band)")
plt.show()
daily_sleep.reset_index().to_csv("src/data/cleaned_sleep_timeseries.csv", index=False)

daily_sleep = pd.read_csv("src/data/cleaned_sleep_timeseries.csv")
daily_sleep["date"] = pd.to_datetime(daily_sleep["index"])
daily_sleep.set_index("date", inplace=True)

# ACF plot to check for weekly/monthly seasonality

plot_acf(daily_sleep["sleep_duration"], lags=40) # Autocorrelation function (ACF) plot to check for seasonality
plt.title("Autocorrelation (ACF) of Daily Sleep Duration")
plt.xlabel("Lag (days)")
plt.ylabel("Autocorrelation")
plt.tight_layout()
plt.show()
# results: There is a very strong spike at lag 1, which is expected (today's sleep duration is correlated with yesterday's).
# beyond lag 1, the autocorrelations remain mild and gradually decay, but stay within or near the confidence interval.
# no strong spikes at lag = 7, 14, 21… → no clear weekly seasonality.