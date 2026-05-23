import os
import warnings
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


DATA_PATH = "data/eur_ron_monthly.csv"
FIG_DIR = "figures"
OUTPUT_DIR = "output"
MIN_DIFFERENCING = 1


@dataclass
class ModelSelection:
    order: Tuple[int, int, int]
    trend: str
    aic: float


def adf_test(series: pd.Series) -> float:
    result = adfuller(series.dropna(), autolag="AIC")
    return result[1]


def difference_until_stationary(series: pd.Series, max_d: int = 2) -> Tuple[pd.Series, int]:
    d = 0
    current = series.copy()
    while d <= max_d:
        p_value = adf_test(current)
        if p_value < 0.05:
            return current, d
        current = current.diff().dropna()
        d += 1
    return current, d


def select_arima_order(series: pd.Series, d: int, p_max: int = 3, q_max: int = 3) -> ModelSelection:
    trend = "t" if d > 0 else "c"
    best = ModelSelection(order=(0, d, 0), trend=trend, aic=np.inf)
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(series, order=(p, d, q), trend=trend)
                result = model.fit(method_kwargs={"maxiter": 200})
                if result.aic < best.aic:
                    best = ModelSelection(order=(p, d, q), trend=trend, aic=result.aic)
            except Exception:
                continue
    return best


def main() -> None:
    warnings.filterwarnings("ignore", message="No frequency information was provided")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge")
    warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
    warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["time"])
    df = df.dropna().sort_values("time")
    df = df.set_index("time").asfreq("MS")
    df = df.dropna()

    series = np.log(df["values"])

    plt.figure(figsize=(10, 4))
    sns.lineplot(x=series.index, y=series.values)
    plt.title("EUR/RON Monthly (log)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "app1_series_log.png"))
    plt.close()

    stationary_series, d = difference_until_stationary(series)
    d = max(d, MIN_DIFFERENCING)
    if d > 0:
        stationary_series = series.diff(d).dropna()
    print(f"Selected differencing order d={d}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(stationary_series, ax=axes[0], lags=24)
    plot_pacf(stationary_series, ax=axes[1], lags=24, method="ywm")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "app1_acf_pacf.png"))
    plt.close()

    train = series.iloc[:-6]
    test = series.iloc[-6:]

    selection = select_arima_order(train, d)
    print(
        f"Best ARIMA order: {selection.order}, trend='{selection.trend}' "
        f"(AIC={selection.aic:.2f})"
    )

    model = ARIMA(train, order=selection.order, trend=selection.trend)
    result = model.fit(method_kwargs={"maxiter": 200})

    resid = result.resid.dropna()
    ljung = acorr_ljungbox(resid, lags=[12], return_df=True)
    jb_stat, jb_p, _, _ = jarque_bera(resid)

    diagnostics = (
        f"Ljung-Box p-value (lag 12): {ljung['lb_pvalue'].iloc[0]:.4f}\n"
        f"Jarque-Bera p-value: {jb_p:.4f}\n"
    )
    with open(os.path.join(OUTPUT_DIR, "app1_diagnostics.txt"), "w", encoding="utf-8") as handle:
        handle.write(diagnostics)

    forecast = result.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    forecast_level = np.exp(forecast_mean)
    ci_lower = np.exp(forecast_ci.iloc[:, 0])
    ci_upper = np.exp(forecast_ci.iloc[:, 1])

    actual_level = np.exp(test)
    mae = np.mean(np.abs(actual_level - forecast_level))
    rmse = np.sqrt(np.mean((actual_level - forecast_level) ** 2))
    mape = np.mean(np.abs((actual_level - forecast_level) / actual_level)) * 100

    metrics = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\n"
    with open(os.path.join(OUTPUT_DIR, "app1_metrics.txt"), "w", encoding="utf-8") as handle:
        handle.write(metrics)

    plt.figure(figsize=(10, 4))
    plt.plot(series.index, np.exp(series), label="Actual")
    plt.plot(test.index, forecast_level, label="Forecast")
    plt.fill_between(test.index, ci_lower, ci_upper, color="gray", alpha=0.2, label="95% CI")
    plt.title("EUR/RON Forecast (level)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "app1_forecast.png"))
    plt.close()


if __name__ == "__main__":
    main()
