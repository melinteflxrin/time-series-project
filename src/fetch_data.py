import os
import re
from typing import Dict, Optional

import pandas as pd
import eurostat

from config import (
    APP2_OUTPUT,
    HICP_MONTHLY_DATASET,
    HICP_MONTHLY_FILTERS,
    HICP_MONTHLY_OUTPUT,
    SHORT_RATE_MONTHLY_DATASET,
    SHORT_RATE_MONTHLY_FILTERS,
    SHORT_RATE_MONTHLY_OUTPUT,
    INDPROD_MONTHLY_DATASET,
    INDPROD_MONTHLY_FILTERS,
    INDPROD_MONTHLY_OUTPUT,
)


def parse_eurostat_time(series: pd.Series) -> pd.DatetimeIndex:
    value = series.astype(str)
    if value.str.contains("M").any():
        return pd.to_datetime(value.str.replace("M", "-", regex=False), format="%Y-%m")
    if value.str.match(r"^\d{4}$").all():
        return pd.to_datetime(value, format="%Y")
    return pd.to_datetime(value, errors="coerce")


def apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, str]]) -> pd.DataFrame:
    if not filters:
        return df
    for key, target in filters.items():
        column = key if key in df.columns else None
        if column is None:
            matches = [col for col in df.columns if key in str(col)]
            if len(matches) == 1:
                column = matches[0]
        if column is None:
            print(f"Warning: filter key '{key}' not found in columns, skipping.")
            continue
        filtered = df[df[column] == target]
        if filtered.empty:
            print(f"Warning: filter '{key}={target}' returns empty dataframe, skipping.")
            continue
        df = filtered
    return df


def normalize_eurostat(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns and "values" in df.columns:
        return df

    time_columns = [col for col in df.columns if re.match(r"^\d{4}(-\d{2})?$", str(col))]
    if time_columns:
        id_columns = [col for col in df.columns if col not in time_columns]
        df = df.melt(
            id_vars=id_columns,
            value_vars=time_columns,
            var_name="time",
            value_name="values",
        )
        return df

    time_candidates = [col for col in df.columns if "TIME_PERIOD" in str(col).upper()]
    if time_candidates and "values" in df.columns:
        df = df.rename(columns={time_candidates[0]: "time"})
        return df

    raise ValueError("Could not normalize Eurostat data frame to time/value format.")


def pick_most_complete_series(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns or "values" not in df.columns:
        raise ValueError("Expected 'time' and 'values' columns.")

    id_columns = [c for c in df.columns if c not in {"time", "values"}]
    if not id_columns:
        return df

    counts = df.groupby(id_columns, dropna=False).size().reset_index(name="count")
    best = counts.sort_values("count", ascending=False).head(1)
    for column in id_columns:
        df = df[df[column] == best.iloc[0][column]]
    return df


def load_series(dataset: str, filters: Optional[Dict[str, str]], output_path: str) -> pd.DataFrame:
    print(f"Downloading {dataset}...")
    df = eurostat.get_data_df(dataset)
    print(f"  Columns: {list(df.columns)}")

    df = apply_filters(df, filters)
    df = normalize_eurostat(df)
    df = pick_most_complete_series(df)

    df = df[["time", "values"]].copy()
    df["time"] = parse_eurostat_time(df["time"])
    df["values"] = pd.to_numeric(df["values"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved {output_path} ({len(df)} rows, {df['time'].min().date()} – {df['time'].max().date()})")
    return df


def main() -> None:
    hicp = load_series(HICP_MONTHLY_DATASET, HICP_MONTHLY_FILTERS, HICP_MONTHLY_OUTPUT)
    rate = load_series(SHORT_RATE_MONTHLY_DATASET, SHORT_RATE_MONTHLY_FILTERS, SHORT_RATE_MONTHLY_OUTPUT)
    indprod = load_series(INDPROD_MONTHLY_DATASET, INDPROD_MONTHLY_FILTERS, INDPROD_MONTHLY_OUTPUT)

    merged = (
        hicp.rename(columns={"values": "hicp"})
        .merge(rate.rename(columns={"values": "short_rate"}), on="time", how="inner")
        .merge(indprod.rename(columns={"values": "indprod"}), on="time", how="inner")
    )
    merged = merged.dropna()

    merged.to_csv(APP2_OUTPUT, index=False)
    print(f"\nSaved {APP2_OUTPUT} ({len(merged)} rows, {merged['time'].min().date()} – {merged['time'].max().date()})")


if __name__ == "__main__":
    main()
