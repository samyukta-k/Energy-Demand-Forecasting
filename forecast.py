from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

TARGET_COL   = "Global_active_power"
FORECAST_HRS = 24          
PLOTS_DIR    = Path("plots")

FEATURE_COLS = [
    "hour", "dayofweek", "month", "is_weekend",
    "hour_sin", "hour_cos",
    "lag_1", "lag_24", "lag_168",
    "rolling_24h", "rolling_168h",
]

def build_history_buffer(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    required_lags: int = 168,
) -> pd.Series:
    
    buffer = df[target_col].copy().sort_index()
    return buffer.iloc[-(required_lags + 1):]

def build_feature_row(
    timestamp: pd.Timestamp,
    history: pd.Series,
) -> pd.DataFrame:
    
    hour      = timestamp.hour
    dayofweek = timestamp.dayofweek
    month     = timestamp.month
    is_weekend = int(dayofweek >= 5)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    def safe_lag(series: pd.Series, n: int) -> float:
        return float(series.iloc[-n]) if len(series) >= n else np.nan

    lag_1   = safe_lag(history, 1)
    lag_24  = safe_lag(history, 24)
    lag_168 = safe_lag(history, 168)

    rolling_24h  = float(history.iloc[-24:].mean())  if len(history) >= 24  else np.nan
    rolling_168h = float(history.iloc[-168:].mean()) if len(history) >= 168 else np.nan

    row = pd.DataFrame([{
        "hour":        hour,
        "dayofweek":   dayofweek,
        "month":       month,
        "is_weekend":  is_weekend,
        "hour_sin":    hour_sin,
        "hour_cos":    hour_cos,
        "lag_1":       lag_1,
        "lag_24":      lag_24,
        "lag_168":     lag_168,
        "rolling_24h": rolling_24h,
        "rolling_168h":rolling_168h,
    }])

    return row[FEATURE_COLS]

def generate_future_forecast(
    model,
    df: pd.DataFrame,
    model_type: str = "xgboost",
    forecast_hours: int = FORECAST_HRS,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    
    last_timestamp = df.index[-1]

    if model_type.lower() == "xgboost":

        history = build_history_buffer(df, target_col=target_col)

        predictions = []
        timestamps = []

        for step in range(1, forecast_hours + 1):

            future_ts = last_timestamp + timedelta(hours=step)

            X_row = build_feature_row(future_ts, history)

            y_hat = float(model.predict(X_row)[0])

            y_hat = max(y_hat, 0.0)

            history = pd.concat([
                history,
                pd.Series([y_hat], index=[future_ts])
            ])

            predictions.append(y_hat)
            timestamps.append(future_ts)

    else:

        forecast = model.forecast(steps=forecast_hours)

        predictions = forecast.values

        timestamps = [
            last_timestamp + timedelta(hours=i)
            for i in range(1, forecast_hours + 1)
        ]

    forecast_df = pd.DataFrame({
        "datetime": timestamps,
        "predicted_energy_demand": predictions,
    })

    return forecast_df

def plot_future_forecast(
    forecast_df: pd.DataFrame,
    df_history: pd.DataFrame,
    target_col: str = TARGET_COL,
    plots_dir: Path = PLOTS_DIR,
    context_hours: int = 72,
) -> None:
    
    os.makedirs(plots_dir, exist_ok=True)

    recent = df_history[target_col].iloc[-context_hours:]

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(
        recent.index, recent.values,
        label=f"Actual (last {context_hours} h)",
        color="#1f77b4", linewidth=1.5,
    )

    ax.plot(
        forecast_df["datetime"],
        forecast_df["predicted_energy_demand"],
        label="24-Hour Forecast",
        color="#ff7f0e", linewidth=2,
        linestyle="--", marker="o", markersize=4,
    )

    ax.axvspan(
        forecast_df["datetime"].iloc[0],
        forecast_df["datetime"].iloc[-1],
        alpha=0.08, color="#ff7f0e", label="Forecast Window",
    )

    ax.axvline(
        x=df_history.index[-1],
        color="grey", linestyle=":", linewidth=1.2, label="Now",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax.set_title(
        "XGBoost Energy Demand Forecast — Next 24 Hours",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = plots_dir / "future_forecast_24h.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def run_forecast_pipeline(
    model,
    df: pd.DataFrame,
    model_type: str = "xgboost",
    forecast_hours: int = FORECAST_HRS,
    plots_dir: Path = PLOTS_DIR,
) -> pd.DataFrame:
    
    forecast_df = generate_future_forecast(
    model=model,
    df=df,
    model_type=model_type,
    forecast_hours=forecast_hours,
)

    plot_future_forecast(forecast_df, df, plots_dir=plots_dir)

    return forecast_df

if __name__ == "__main__":
    import sys
    from preprocess import run_pipeline
    from xgboost_model import run_xgboost_pipeline
    from statistical_models import (
        chronological_split,
        run_arima,
        run_sarimax,
        compare_models,
)

    data_path = (
        sys.argv[1] if len(sys.argv) > 1 else "household_power_consumption.txt"
    )

    df_processed = run_pipeline(data_path)

train_df, test_df = chronological_split(df_processed)

xgb_results = run_xgboost_pipeline(df_processed)

arima_results = run_arima(train_df, test_df)

sarimax_results = run_sarimax(train_df, test_df)

comparison, best_model = compare_models(
    arima_results["metrics"],
    sarimax_results["metrics"],
    xgb_results["metrics"],
)

if best_model == "XGBoost":

    selected_model = xgb_results["model"]

    forecast = run_forecast_pipeline(
        selected_model,
        df_processed,
        model_type="xgboost",
    )

elif best_model == "SARIMAX":

    selected_model = sarimax_results["model"]

    forecast = run_forecast_pipeline(
        selected_model,
        df_processed,
        model_type="sarimax",
    )

else:

    selected_model = arima_results["model"]

    forecast = run_forecast_pipeline(
        selected_model,
        df_processed,
        model_type="arima",
    )
