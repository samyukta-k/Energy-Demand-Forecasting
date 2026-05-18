import numpy as np
import pandas as pd

def load_raw_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=";",
        na_values=["?", "NA"],
        low_memory=False,
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        format="%d/%m/%Y %H:%M:%S",
    )

    df = (
        df.drop(columns=["Date", "Time"])
          .set_index("datetime")
          .sort_index()
    )
    return df

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df_hour = df.resample("h").mean()

    df_hour = df_hour.interpolate(method="time")

    df_hour = df_hour.ffill().bfill()

    return df_hour

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek          
    df["month"]      = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)  
    return df

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    lags: list[int] = None,
) -> pd.DataFrame:
    
    if lags is None:
        lags = [1, 24, 168]

    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    windows: list[int] = None,
) -> pd.DataFrame:
    
    if windows is None:
        windows = [24, 168]

    df = df.copy()
    for w in windows:
        df[f"rolling_{w}h"] = df[target_col].rolling(window=w).mean()
    return df


def add_cyclical_features(
    df: pd.DataFrame,
    col: str = "hour",
    period: int = 24,
) -> pd.DataFrame:
    
    df = df.copy()
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df

def run_pipeline(
    filepath: str,
    target_col: str = "Global_active_power",
    lag_offsets: list[int] = None,
    rolling_windows: list[int] = None,
) -> pd.DataFrame:
    
    if lag_offsets is None:
        lag_offsets = [1, 24, 168]
    if rolling_windows is None:
        rolling_windows = [24, 168]

    df = load_raw_data(filepath)

    df = convert_numeric_columns(df)

    df = resample_to_hourly(df)

    df = add_time_features(df)

    df = add_lag_features(df, target_col=target_col, lags=lag_offsets)
    df = add_rolling_features(df, target_col=target_col, windows=rolling_windows)

    df = add_cyclical_features(df, col="hour", period=24)

    df = df.dropna()

    return df

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "household_power_consumption.txt"
    df_processed = run_pipeline(path)
    print(df_processed.head())
    print(df_processed.columns.tolist())