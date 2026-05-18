from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

TARGET_COL = "Global_active_power"

PLOTS_DIR = Path("plots")

def compute_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    nonzero_mask = y_true != 0

    mape = (
        np.mean(
            np.abs(
                (y_true[nonzero_mask] - y_pred[nonzero_mask])
                / y_true[nonzero_mask]
            )
        )
        * 100
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }

def chronological_split(df, test_fraction=0.2):

    split_idx = int(len(df) * (1 - test_fraction))

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    return train, test

def run_arima(train, test):

    model = ARIMA(
        train[TARGET_COL],
        order=(3, 1, 3)
    )

    fitted = model.fit()

    forecast = fitted.forecast(steps=len(test))

    metrics = compute_metrics(
        test[TARGET_COL].values,
        forecast.values
    )

    return {
        "model": fitted,
        "forecast": forecast,
        "metrics": metrics,
    }

def run_sarimax(train, test):

    exog_cols = [
        "hour",
        "dayofweek",
        "is_weekend",
        "lag_24",
        "rolling_24h",
    ]

    model = SARIMAX(
        train[TARGET_COL],
        exog=train[exog_cols],
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(disp=False)

    forecast = fitted.forecast(
        steps=len(test),
        exog=test[exog_cols]
    )

    metrics = compute_metrics(
        test[TARGET_COL].values,
        forecast.values
    )

    return {
        "model": fitted,
        "forecast": forecast,
        "metrics": metrics,
    }

def plot_model_comparison(
    test,
    arima_forecast,
    sarimax_forecast,
    xgb_forecast,
):

    import os

    os.makedirs(PLOTS_DIR, exist_ok=True)

    idx = test.index[-24 * 14:]

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(
        idx,
        test[TARGET_COL].iloc[-24 * 14:],
        label="Actual",
        linewidth=2,
    )

    ax.plot(
        idx,
        arima_forecast[-24 * 14:],
        label="ARIMA",
        linestyle="--",
    )

    ax.plot(
        idx,
        sarimax_forecast[-24 * 14:],
        label="SARIMAX",
        linestyle="--",
    )

    ax.plot(
        idx,
        xgb_forecast[-24 * 14:],
        label="XGBoost",
        linestyle="--",
    )

    ax.legend()

    ax.set_title("Model Comparison")

    ax.grid(True, linestyle=":")

    fig.tight_layout()

    out_path = PLOTS_DIR / "model_comparison.png"

    fig.savefig(out_path, dpi=150)

    plt.close(fig)

def compare_models(arima_metrics, sarimax_metrics, xgb_metrics):

    comparison = pd.DataFrame({
        "Model": ["ARIMA", "SARIMAX", "XGBoost"],
        "MAE": [
            arima_metrics["mae"],
            sarimax_metrics["mae"],
            xgb_metrics["mae"],
        ],
        "RMSE": [
            arima_metrics["rmse"],
            sarimax_metrics["rmse"],
            xgb_metrics["rmse"],
        ],
        "MAPE": [
            arima_metrics["mape"],
            sarimax_metrics["mape"],
            xgb_metrics["mape"],
        ],
    })

    print("MODEL COMPARISON")

    print(comparison)

    best_model = comparison.sort_values("RMSE").iloc[0]["Model"]

    print(f"\nBest Model: {best_model}")

    return comparison, best_model