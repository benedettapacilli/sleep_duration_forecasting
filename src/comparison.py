import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, wilcoxon

arima_df = pd.read_csv("src/results/arima_forecast.csv")
mlp_df = pd.read_csv("src/results/mlp_forecast.csv")
rf_df = pd.read_csv("src/results/rf_forecast.csv")
true_df = pd.read_csv("src/data/cleaned_sleep_timeseries.csv")
true_df["date"] = pd.to_datetime(true_df["index"])
true_df.set_index("date", inplace=True)

arima_dates = set(pd.to_datetime(arima_df["date"]))
mlp_dates = set(pd.to_datetime(mlp_df["date"]))
rf_dates = set(pd.to_datetime(rf_df["date"]))
common_dates = sorted(arima_dates & mlp_dates & rf_dates) # intersection of all forecast dates
all_dates = pd.to_datetime(common_dates)
true_values = true_df.loc[all_dates]["sleep_duration"].values # aligning range of true values with forecast dates

plt.figure(figsize=(12, 6))
plt.plot(all_dates, true_values, label="Actual", linewidth=2)
plt.plot(pd.to_datetime(arima_df["date"]), arima_df["prediction"], label="ARIMA", linestyle="--")
plt.plot(pd.to_datetime(mlp_df["date"]), mlp_df["prediction"], label="MLP", linestyle="--")
plt.plot(pd.to_datetime(rf_df["date"]), rf_df["prediction"], label="Random Forest", linestyle="--")
plt.title("Forecast Comparison: ARIMA vs MLP vs RF")
plt.xlabel("Date")
plt.ylabel("Sleep Duration (minutes)")
plt.legend()
plt.tight_layout()
plt.show()

# extract predictions and true values for error analysis
arima_preds = arima_df.set_index(pd.to_datetime(arima_df["date"])).loc[all_dates]["prediction"].values
mlp_preds = mlp_df.set_index(pd.to_datetime(mlp_df["date"])).loc[all_dates]["prediction"].values
rf_preds = rf_df.set_index(pd.to_datetime(rf_df["date"])).loc[all_dates]["prediction"].values
errors_arima = arima_preds - true_values
errors_mlp = mlp_preds - true_values
errors_rf = rf_preds - true_values

# Shapiro-Wilk normality test to check if residuals are normally distributed, meaning the errors should be normally distributed for the models to be comparable
print("\nShapiro-Wilk Normality Test (on residuals):")
for name, errors in zip(["ARIMA", "MLP", "RF"], [errors_arima, errors_mlp, errors_rf]):
    stat, p = shapiro(errors)
    print(f"{name} Errors: p = {p:.4f} {'Not normal' if p < 0.05 else 'Normal'}") # results: all three models have residuals that pass the normality test (p > 0.05), meaning their prediction errors are approximately normally distributed.

# Wilcoxon signed-rank tests (paired, non-parametric)
print("\nWilcoxon Signed-Rank Tests (pairwise):")
p1 = wilcoxon(errors_arima, errors_mlp).pvalue
print(f"ARIMA vs MLP: p = {p1:.4f}")
p2 = wilcoxon(errors_arima, errors_rf).pvalue
print(f"ARIMA vs RF:  p = {p2:.4f}")
p3 = wilcoxon(errors_mlp, errors_rf).pvalue
print(f"MLP vs RF:    p = {p3:.4f}")
# results: all pairwise model comparisons are statistically significant (p < 0.05), indicating meaningful differences in forecasting performance between them