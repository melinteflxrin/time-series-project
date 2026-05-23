import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, coint_johansen


DATA_PATH = "data/app2_monthly.csv"
FIG_DIR = "figures"
OUTPUT_DIR = "output"


def adf_pvalue(series: pd.Series) -> float:
    return adfuller(series.dropna(), autolag="AIC")[1]


def print_adf_report(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        p_val = adf_pvalue(df[col])
        print(f"ADF p-value for {col}: {p_val:.4f}")


def granger_report(df: pd.DataFrame, max_lag: int = 6) -> None:
    for y in df.columns:
        for x in df.columns:
            if x == y:
                continue
            print(f"\nGranger causality: {x} -> {y}")
            grangercausalitytests(df[[y, x]], maxlag=max_lag, verbose=True)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["time"]).dropna().sort_values("time")
    df = df.set_index("time")

    columns = ["eur_ron", "hicp", "short_rate"]
    df = df[columns]

    print_adf_report(df, columns)

    johansen = coint_johansen(df, det_order=0, k_ar_diff=2)
    trace_stat = johansen.lr1
    crit = johansen.cvt[:, 1]
    coint_rank = int(np.sum(trace_stat > crit))
    print(f"Cointegration rank (5%): {coint_rank}")

    if coint_rank > 0:
        lag_select = select_order(df, maxlags=6, deterministic="co")
        lag_order = lag_select.aic
        model = VECM(df, k_ar_diff=lag_order, coint_rank=coint_rank, deterministic="co")
        result = model.fit()
        with open(os.path.join(OUTPUT_DIR, "app2_vecm_summary.txt"), "w", encoding="utf-8") as handle:
            handle.write(result.summary().as_text())
        irf = result.irf(12)
    else:
        diff_df = df.diff().dropna()
        lag_order = VAR(diff_df).select_order(6).aic
        model = VAR(diff_df)
        result = model.fit(lag_order)
        with open(os.path.join(OUTPUT_DIR, "app2_var_summary.txt"), "w", encoding="utf-8") as handle:
            handle.write(result.summary().as_text())
        irf = result.irf(12)

    irf.plot(orth=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "app2_irf.png"))
    plt.close()

    granger_report(df)


if __name__ == "__main__":
    main()
