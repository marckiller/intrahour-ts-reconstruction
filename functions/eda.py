import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_parquet(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def summarize_dataframe(df):
    if df is None:
        print("DataFrame is None. Unable to summarize.")
        return

    print("\nDataFrame Summary")
    print("=" * 40)
    print(f"Shape: {df.shape}")
    print("\nColumns and Data Types:")
    print(df.dtypes)

def plot_random_time_series(df, column, nrows=2, ncols=3, figsize=(12, 8)):
    if df is None or column not in df.columns:
        print(f"DataFrame must contain the specified column: {column}")
        return
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    sampled_df = df.sample(min(nrows * ncols, len(df)), random_state=42)
    
    for i, (_, row) in enumerate(sampled_df.iterrows()):
        ax = axes[i]
        if isinstance(row[column], (list, np.ndarray)):
            time_series = row[column]
            ticker = 'WIG20' if column == 'series_index' else row.get('ticker', 'Unknown')
        else:
            continue
        
        timestamp = row.get('timestamp', 'Unknown')
        ax.plot(time_series)
        ax.set_title(f"{ticker} - {timestamp}")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def filter_by_hour(df, hour_column='timestamp', cutoff_hour=16):
    if hour_column not in df.columns:
        print(f"Column '{hour_column}' not found in DataFrame.")
        return df

    df_filtered = df[pd.to_datetime(df[hour_column]).dt.hour < cutoff_hour]
    print(f"Filtered DataFrame: {df_filtered.shape[0]} rows remaining.")
    return df_filtered

def filter_non_empty_series(df, series_column='series'):
    if series_column not in df.columns:
        print(f"Column '{series_column}' not found in DataFrame.")
        return df

    df_filtered = df[df[series_column].apply(lambda x: isinstance(x, (list, np.ndarray)) and np.any(pd.notna(x)))]
    print(f"Filtered DataFrame: {df_filtered.shape[0]} rows remaining.")
    return df_filtered

def filter_by_completeness(df, series_column='series', min_values=5):
    if series_column not in df.columns:
        print(f"Column '{series_column}' not found in DataFrame.")
        return df

    df_filtered = df[df[series_column].apply(lambda x: isinstance(x, (list, np.ndarray)) and np.sum(pd.notna(x)) >= min_values)]
    print(f"Filtered DataFrame: {df_filtered.shape[0]} rows remaining (min {min_values} valid values).")
    return df_filtered

import os

def save_ml_ready(df, filename="ml_ready.parquet", folder="data/processed"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    
    try:
        df.to_parquet(file_path, index=False)
        print(f"Saved ML-ready dataset to {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")