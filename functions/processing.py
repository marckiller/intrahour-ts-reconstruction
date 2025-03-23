import pandas as pd
import numpy as np

def create_timestamp_column(df):

    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + df["time"].astype(str).str.zfill(6), 
        format='%Y%m%d%H%M%S'
    )
    return df

def resample_ohlc(df, frequency):

    resampled = df.resample(frequency).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    return resampled

def attach_minute_series(df_hour, df_minute):

    minute_bins = [minute.minute for minute in pd.date_range(start="00:00:00", end="00:59:00", freq="1min")]

    minute_series = []

    for hour in df_hour.index:
        minute_data = df_minute.loc[hour:hour + pd.Timedelta(minutes=59)]

        if not minute_data.empty:
            minute_data = minute_data.reindex(hour + pd.to_timedelta(minute_bins, unit="m")).fillna(np.nan)
            minute_series.append(minute_data["close"].to_list())

        else:
            minute_series.append([np.nan] * len(minute_bins))

    df_hour["series"] = minute_series
    return df_hour

def add_index_series(df_hour_instrument, df_hour_index):

    df_hour_instrument, df_hour_index = df_hour_instrument.align(df_hour_index, join='inner')
    df_hour_instrument['series_index'] = df_hour_index['series']
    
    return df_hour_instrument

def calculate_rolling_correlations(df_instrument, df_index, windows, window_unit = "h"):

    for window in windows:
        column_name = f"corr_{window}{window_unit}"
        df_instrument[column_name] = (
            df_instrument["return"]
            .rolling(window=window, min_periods=1)
            .corr(df_index["return"])
        )
    
    return df_instrument

def merge_daily_into_hourly(df_hour_instrument, df_day_instrument):

    overlapping_columns = set(df_hour_instrument.columns) & set(df_day_instrument.columns)
    additional_columns = [col for col in df_day_instrument.columns if col not in overlapping_columns]
    
    for col in additional_columns:
        df_hour_instrument[col] = df_hour_instrument.index.floor('D').map(df_day_instrument[col])
    
    return df_hour_instrument

def add_index_ohlc(df_instrument, df_index):
    
    df_index_renamed = df_index[["open", "high", "low", "close"]].add_suffix("_index")
    df_instrument = df_instrument.join(df_index_renamed, how="left")
    return df_instrument

def merge_instrument_data(dataset, df_instrument_hour, ticker_name):

    df = df_instrument_hour.copy()
    
    df = df.reset_index()
    
    df["ticker"] = ticker_name

    if dataset.empty:
        return df
    else:
        common_cols = dataset.columns.intersection(df.columns)
        return pd.concat([dataset[common_cols], df[common_cols]], ignore_index=True)