from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

TARGET_COL = "Global_active_power"

FEATURE_COLS = [
    "hour", "dayofweek", "month", "is_weekend",
    "hour_sin", "hour_cos",
    "lag_1", "lag_24", "lag_168",
    "rolling_24h", "rolling_168h",
]

PLOTS_DIR = Path("plots")

DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,       
    "reg_lambda": 1.0,      
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

def chronological_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    feature_cols: list[str] = None,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        None

    split_idx = int(len(df) * (1 - test_fraction))

    X = df[available]
    y = df[target_col]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(
        f"Train: {X_train.index[0].date()} to {X_train.index[-1].date()}  "
        f"({len(X_train):,} rows)\n"
        f"Test : {X_test.index[0].date()} to {X_test.index[-1].date()}  "
        f"({len(X_test):,} rows)"
    )
    return X_train, X_test, y_train, y_test

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    xgb_params: dict = None,
) -> XGBRegressor:
    
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    model = XGBRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )


    return model

def generate_predictions(
    model: XGBRegressor,
    X_test: pd.DataFrame,
) -> np.ndarray:
    
    return model.predict(X_test)

def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    nonzero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask])
                          / y_true[nonzero_mask])) * 100

    return {"mae": mae, "rmse": rmse, "mape": mape}


def print_metrics(metrics: dict[str, float], model_name: str = "XGBoost") -> None:
    print(f"  {model_name} Model Performance")
    print(f"  MAE  : {metrics['mae']:.4f}  kW")
    print(f"  RMSE : {metrics['rmse']:.4f}  kW")
    print(f"  MAPE : {metrics['mape']:.2f} %")

def plot_forecast_vs_actual(
    y_test: pd.Series,
    y_pred: np.ndarray,
    plots_dir: Path = PLOTS_DIR,
    last_n_days: int = 14,
) -> None:
    
    os.makedirs(plots_dir, exist_ok=True)

    n_hours = last_n_days * 24
    idx    = y_test.index[-n_hours:]
    actual = y_test.values[-n_hours:]
    pred   = y_pred[-n_hours:]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(idx, actual, label="Actual",   linewidth=1.5, color="#1f77b4")
    ax.plot(idx, pred,   label="Forecast", linewidth=1.5, color="#ff7f0e",
            linestyle="--", alpha=0.85)

    ax.set_title(
        f"XGBoost Forecast vs Actual: Last {last_n_days} Days",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = plots_dir / "forecast_vs_actual.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    model: XGBRegressor,
    feature_names: list[str],
    plots_dir: Path = PLOTS_DIR,
    top_n: int = 15,
) -> None:
    
    os.makedirs(plots_dir, exist_ok=True)

    importances = model.feature_importances_
    importance_df = (
        pd.Series(importances, index=feature_names)
          .sort_values(ascending=True)
          .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    bars = ax.barh(
        importance_df.index,
        importance_df.values,
        color="#2196F3",
        edgecolor="white",
        height=0.65,
    )

    for bar, val in zip(bars, importance_df.values):
        ax.text(
            bar.get_width() + importance_df.values.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=8, color="#333333",
        )

    ax.set_title(f"XGBoost Feature Importance: Top {top_n} (Gain)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Gain")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out_path = plots_dir / "feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def run_xgboost_pipeline(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    feature_cols: list[str] = None,
    test_fraction: float = 0.2,
    xgb_params: dict = None,
    plots_dir: Path = PLOTS_DIR,
) -> dict:
    
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_train, X_test, y_train, y_test = chronological_split(
        df, target_col=target_col, feature_cols=feature_cols,
        test_fraction=test_fraction,
    )

    model = train_xgboost(X_train, y_train, xgb_params=xgb_params)

    y_pred  = generate_predictions(model, X_test)
    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics)

    plot_forecast_vs_actual(y_test, y_pred, plots_dir=plots_dir)
    plot_feature_importance(model, X_test.columns.tolist(), plots_dir=plots_dir)

    return {
        "model":   model,
        "metrics": metrics,
        "y_test":  y_test,
        "y_pred":  y_pred,
        "X_test":  X_test,
    }

if __name__ == "__main__":
    import sys
    from preprocess import run_pipeline  

    data_path = (
        sys.argv[1] if len(sys.argv) > 1 else "household_power_consumption.txt"
    )

    df_processed = run_pipeline(data_path)

    results = run_xgboost_pipeline(df_processed)