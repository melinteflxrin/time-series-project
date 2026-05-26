import os
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, kpss


DATA_PATH = "../data/hicp_ro_monthly.csv"
FIG_DIR = "../figures"
OUTPUT_DIR = "../output"
SEASONAL_PERIOD = 12
FORECAST_STEPS = 12


@dataclass
class ModelSelection:
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    aic: float


def adf_pvalue(series: pd.Series) -> float:
    return adfuller(series.dropna(), autolag="AIC")[1]


def kpss_pvalue(series: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kpss(series.dropna(), regression="c", nlags="auto")[1]


def determine_d(series: pd.Series) -> int:
    adf_p = adf_pvalue(series)
    kpss_p = kpss_pvalue(series)
    # d=0 only when both tests agree the series is stationary
    if adf_p < 0.05 and kpss_p >= 0.05:
        return 0
    if adf_pvalue(series.diff().dropna()) < 0.05:
        return 1
    return 2


def determine_D(differenced: pd.Series) -> int:
    """Return 1 if the ACF at lag 12 is significant, suggesting seasonal non-stationarity."""
    n = len(differenced.dropna())
    threshold = 1.96 / np.sqrt(n)
    acf_vals = acf(differenced.dropna(), nlags=SEASONAL_PERIOD, fft=True)
    return 1 if abs(acf_vals[SEASONAL_PERIOD]) > threshold else 0


def apply_diff(s: pd.Series, d: int, D: int, s_period: int) -> pd.Series:
    result = s.copy()
    for _ in range(d):
        result = result.diff()
    for _ in range(D):
        result = result.diff(s_period)
    return result.dropna()


def select_sarima(series: pd.Series, d: int, D: int) -> ModelSelection:
    best = ModelSelection(
        order=(1, d, 1),
        seasonal_order=(1, D, 1, SEASONAL_PERIOD),
        aic=np.inf,
    )
    for p, q, P, Q in product(range(3), range(3), range(2), range(2)):
        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue
        try:
            fit = ARIMA(
                series,
                order=(p, d, q),
                seasonal_order=(P, D, Q, SEASONAL_PERIOD),
            ).fit(method_kwargs={"maxiter": 300})
            if fit.aic < best.aic:
                best = ModelSelection(
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, SEASONAL_PERIOD),
                    aic=fit.aic,
                )
        except Exception:
            continue
    return best


def main() -> None:
    warnings.filterwarnings("ignore")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["time"]).dropna().sort_values("time")
    df = df.set_index("time").asfreq("MS").dropna()
    series = np.log(df["values"])

    # ── 1. Plot original log series ──────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    sns.lineplot(x=series.index, y=series.values)
    plt.title("Romania HICP – log(index 2015=100), monthly")
    plt.xlabel("Date")
    plt.ylabel("log(HICP)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "app1_series_log.png"), dpi=150)
    plt.close()

    # ── 2. Unit root tests ───────────────────────────────────────────────────
    adf_level = adf_pvalue(series)
    kpss_level = kpss_pvalue(series)
    d = determine_d(series)

    # series.diff(0) returns all-zeros, so treat d=0 as "already stationary"
    diff_series = series.diff(d).dropna() if d > 0 else series.copy()
    adf_diff = adf_level if d == 0 else adf_pvalue(diff_series)

    D = determine_D(diff_series)
    stationary = diff_series.diff(SEASONAL_PERIOD).dropna() if D == 1 else diff_series

    print(f"ADF p-value (level): {adf_level:.4f}")
    print(f"KPSS p-value (level): {kpss_level:.4f}")
    print(f"ADF p-value (diff d={d}): {adf_diff:.4f}")
    print(f"Selected: d={d}, D={D}")

    # ── 3. ACF / PACF of stationary series ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(stationary, ax=axes[0], lags=48)
    plot_pacf(stationary, ax=axes[1], lags=48, method="ywm")
    axes[0].set_title(f"ACF  (d={d}, D={D})")
    axes[1].set_title(f"PACF (d={d}, D={D})")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "app1_acf_pacf.png"), dpi=150)
    plt.close()

    # ── 4. Train / test split ────────────────────────────────────────────────
    train = series.iloc[:-FORECAST_STEPS]
    test = series.iloc[-FORECAST_STEPS:]

    # ── 5. Model selection ───────────────────────────────────────────────────
    print("Searching SARIMA orders (this may take a minute)...")
    selection = select_sarima(train, d, D)
    print(
        f"Best SARIMA{selection.order}x{selection.seasonal_order}  "
        f"AIC={selection.aic:.2f}"
    )

    # ── 6. Fit final model ───────────────────────────────────────────────────
    result = ARIMA(
        train,
        order=selection.order,
        seasonal_order=selection.seasonal_order,
    ).fit(method_kwargs={"maxiter": 300})

    # ── 7. Diagnostics ───────────────────────────────────────────────────────
    resid = result.resid.dropna()
    lb = acorr_ljungbox(resid, lags=[12, 24], return_df=True)
    _, jb_p, _, _ = jarque_bera(resid)
    adf_resid = adf_pvalue(pd.Series(resid.values))

    diagnostics = (
        f"Model: SARIMA{selection.order}x{selection.seasonal_order}\n"
        f"AIC: {selection.aic:.4f}\n"
        f"ADF p-value (level):      {adf_level:.4f}\n"
        f"KPSS p-value (level):     {kpss_level:.4f}\n"
        f"ADF p-value (diff d={d}): {adf_diff:.4f}\n"
        f"Ljung-Box p-value (lag 12): {lb['lb_pvalue'].iloc[0]:.4f}\n"
        f"Ljung-Box p-value (lag 24): {lb['lb_pvalue'].iloc[1]:.4f}\n"
        f"Jarque-Bera p-value:        {jb_p:.4f}\n"
        f"ADF p-value (residuals):    {adf_resid:.4f}\n"
    )
    print(diagnostics)
    with open(os.path.join(OUTPUT_DIR, "app1_diagnostics.txt"), "w", encoding="utf-8") as fh:
        fh.write(diagnostics)

    # ── 8. Forecast ──────────────────────────────────────────────────────────
    fc = result.get_forecast(steps=FORECAST_STEPS)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int()

    fc_level = np.exp(fc_mean)
    ci_lo = np.exp(fc_ci.iloc[:, 0])
    ci_hi = np.exp(fc_ci.iloc[:, 1])
    test_level = np.exp(test)

    mae = np.mean(np.abs(test_level - fc_level))
    rmse = np.sqrt(np.mean((test_level - fc_level) ** 2))
    mape = np.mean(np.abs((test_level - fc_level) / test_level)) * 100

    metrics = f"Forecast horizon: {FORECAST_STEPS} months\nMAE:  {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%\n"
    print(metrics)
    with open(os.path.join(OUTPUT_DIR, "app1_metrics.txt"), "w", encoding="utf-8") as fh:
        fh.write(metrics)

    # ── 9. Stationary-series forecast ────────────────────────────────────────
    # Transform level forecasts into the stationary (differenced) space by
    # anchoring on enough training history and applying the same differencing.
    anchor_len = d + SEASONAL_PERIOD * D + 1
    anchor = train.iloc[-anchor_len:]

    fc_stat = apply_diff(pd.Series(pd.concat([anchor, fc_mean])), d, D, SEASONAL_PERIOD).iloc[-FORECAST_STEPS:]
    fc_stat_lo = apply_diff(pd.Series(pd.concat([anchor, fc_ci.iloc[:, 0]])), d, D, SEASONAL_PERIOD).iloc[-FORECAST_STEPS:]
    fc_stat_hi = apply_diff(pd.Series(pd.concat([anchor, fc_ci.iloc[:, 1]])), d, D, SEASONAL_PERIOD).iloc[-FORECAST_STEPS:]

    stat_history = apply_diff(series, d, D, SEASONAL_PERIOD)
    train_stat = apply_diff(train, d, D, SEASONAL_PERIOD)
    test_stat = stat_history.iloc[-FORECAST_STEPS:]

    mae_stat = float(np.mean(np.abs(test_stat.values - fc_stat.values)))
    rmse_stat = float(np.sqrt(np.mean((test_stat.values - fc_stat.values) ** 2)))

    stat_metrics = (
        f"\nStationary series (d={d}, D={D}) forecast:\n"
        f"MAE:  {mae_stat:.6f}\nRMSE: {rmse_stat:.6f}\n"
    )
    print(stat_metrics)
    with open(os.path.join(OUTPUT_DIR, "app1_metrics.txt"), "a", encoding="utf-8") as fh:
        fh.write(stat_metrics)

    diff_label = (("Seasonal diff of " if D > 0 else "") + ("First diff of " if d > 0 else "") + "log(HICP)") or "log(HICP)"
    plt.figure(figsize=(10, 4))
    plt.plot(train_stat.index, train_stat.values, label="Stationary series", color="steelblue", linewidth=1.0, alpha=0.8)
    bridge_stat_idx = [train_stat.index[-1]] + list(fc_stat.index)
    bridge_stat_vals = [float(train_stat.iloc[-1])] + list(fc_stat.values)
    plt.plot(bridge_stat_idx, bridge_stat_vals, label="Forecast", color="darkorange", linewidth=1.5)
    plt.fill_between(fc_stat.index, fc_stat_lo.values, fc_stat_hi.values, alpha=0.25, color="orange", label="95% CI")
    plt.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    plt.title(f"Stationary HICP Forecast – SARIMA{selection.order}x{selection.seasonal_order}")
    plt.xlabel("Date")
    plt.ylabel(diff_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "app1_stationary_forecast.png"), dpi=150)
    plt.close()

    # ── 10. Level forecast plot ───────────────────────────────────────────────
    train_level = np.exp(train)
    plt.figure(figsize=(10, 4))
    plt.plot(train_level.index, train_level, label="Actual", color="steelblue", linewidth=1.2)
    # Bridge the last actual point into the forecast so the lines visually connect
    bridge_idx = [train_level.index[-1]] + list(fc_level.index)
    bridge_vals = [train_level.iloc[-1]] + list(fc_level.values)
    plt.plot(bridge_idx, bridge_vals, label="Forecast", color="darkorange", linewidth=1.5)
    plt.fill_between(fc_level.index, ci_lo, ci_hi, alpha=0.25, color="orange", label="95% CI")
    plt.title(
        f"Romania HICP Forecast – SARIMA{selection.order}x{selection.seasonal_order}"
    )
    plt.xlabel("Date")
    plt.ylabel("HICP Index (2015=100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "app1_forecast.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
