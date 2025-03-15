import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.processing import *

from config_loader import load_config

config = load_config()

dataset = pd.DataFrame()

raw_data_path = config["raw_data"]["path"]
ticker_index = config["raw_data"]["ticker_index"]
file_extension = config["raw_data"]["file_extension"]

windows = config["preprocessing"]["windows"]

index_file = f"{ticker_index}.{file_extension}"
index_path = f"{raw_data_path}/{index_file}"

instrument_filenames = [f for f in os.listdir(raw_data_path) if not f.startswith(".") and f != index_file]

df_index_complete = pd.read_csv(index_path, header=None, names=["ticker", "unknown1", "date", "time", "open", "high", "low", "close", "volume", "unknown2"])
df_index_complete = create_timestamp_column(df_index_complete)
df_index_complete = df_index_complete.set_index("timestamp")
df_index_complete.drop(columns=["ticker", "unknown1", "unknown2", "date", "time", "volume"], inplace=True)

df_index_complete_min = resample_ohlc(df_index_complete, "min")
df_index_complete_hour = resample_ohlc(df_index_complete_min, "h")



for file in instrument_filenames:
    print(f"Processing {file}...")

    path_instrument = f"{raw_data_path}/{file}"

    df_instrument = pd.read_csv(path_instrument, header=None, names=["ticker", "unknown1", "date", "time", "open", "high", "low", "close", "volume", "unknown2"])
    df_instrument = create_timestamp_column(df_instrument)
    df_instrument = df_instrument.set_index("timestamp")
    ticker_name = df_instrument["ticker"].unique()[0]
    df_instrument.drop(columns=["ticker", "unknown1", "unknown2", "date", "time", "volume"], inplace=True)

    df_instrument_min = resample_ohlc(df_instrument, "min")
    df_instrument_hour = resample_ohlc(df_instrument_min, "h")

    # aligning indexes
    common_index_hour = df_index_complete_hour.index.intersection(df_instrument_hour.index)

    df_index_hour = df_index_complete_hour.loc[common_index_hour]
    df_instrument_hour = df_instrument_hour.loc[common_index_hour]

    # calculate returns
    df_index_hour["return"] = df_index_hour["close"].pct_change()
    df_instrument_hour["return"] = df_instrument_hour["close"].pct_change()

    # attach 60-element minute series to each hour series
    df_index_hour = attach_minute_series(df_index_hour, df_index_complete_min)
    df_instrument_hour = attach_minute_series(df_instrument_hour, df_instrument_min)

    # attach index series to instrument data 
    df_instrument_hour = add_index_series(df_instrument_hour, df_index_hour)

    # calculate rolling statistics
    df_instrument_hour = calculate_rolling_correlations(df_instrument_hour, df_index_hour, windows, "h")

    df_instrument_hour.dropna(inplace=True)

    dataset = merge_instrument_data(dataset, df_instrument_hour, ticker_name)

    print(f"Processed {file}. Dataset shape: {dataset.shape}")

output_dir = config["preprocessing"]["output_path"]
file_name = config["preprocessing"]["output_filename"]

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, file_name)

dataset.to_parquet(output_path, index=False)

print(f"Dataset saved as: {output_path}")
