import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import functions.eda as eda

def normalize_series_column(df, col_series='series', col_mask='series_mask',
                            col_low='low', col_high='high'):
    normalized_series = []

    for s, mask, low, high in zip(df[col_series], df[col_mask], df[col_low], df[col_high]):
        if high == low or not isinstance(s, np.ndarray) or not isinstance(mask, np.ndarray):
            normalized_series.append(np.zeros(60, dtype=np.float32))
            continue

        s_safe = np.nan_to_num(s, nan=0.0)
        mask = mask.astype(bool)

        norm = (s_safe - low) / (high - low)
        norm = np.clip(norm, 0.0, 1.0)
        norm[~mask] = 0.0

        normalized_series.append(norm.astype(np.float32))

    df[col_series] = normalized_series
    return df

def normalize_scalar_column(df, col_value='open', col_low='low', col_high='high'):
    values = df[col_value].astype(float)
    lows = df[col_low].astype(float)
    highs = df[col_high].astype(float)

    denom = highs - lows
    denom = denom.replace(0, np.nan)

    normalized = (values - lows) / denom
    normalized = normalized.clip(lower=0.0, upper=1.0).fillna(0.0)

    df[col_value] = normalized
    return df

def load_filtered_dataset(path, start_date, end_date):
    df = eda.load_parquet(path)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    df['series_mask'] = df['series'].apply(lambda x: np.where(np.isnan(x), 0, 1).astype(np.float32))
    df['series_index_mask'] = df['series_index'].apply(lambda x: np.where(np.isnan(x), 0, 1).astype(np.float32))

    df['series'] = df['series'].apply(lambda x: np.where(np.isnan(x), 0.0, x).astype(np.float32))
    df['series_index'] = df['series_index'].apply(lambda x: np.where(np.isnan(x), 0.0, x).astype(np.float32))

    df = normalize_series_column(df)
    df = normalize_series_column(df, col_series='series_index', col_mask='series_index_mask', col_low='low_index', col_high='high_index')

    for col in ['open', 'close']:
        df = normalize_scalar_column(df, col_value=col, col_low='low', col_high='high')
    for col in ['open_index', 'close_index']:
        df = normalize_scalar_column(df, col_value=col, col_low='low_index', col_high='high_index')

    return df

class FinancialDataset(Dataset):

    def __init__(self, data_path, start_date, end_date, series_mask_fraction = 0.8):
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.mask_series_fraction = series_mask_fraction

        self.df = load_filtered_dataset(self.data_path, self.start_date, self.end_date)
        self.df = self.df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].copy()

        series_target = row['series'].copy()
        series_target_mask = row['series_mask'].copy()

        series = series_target.copy()
        mask = series_target_mask.copy()

        if self.mask_series_fraction > 0:
            indices = np.where(mask == 1)[0]
            num_to_mask = int(len(indices) * self.mask_series_fraction)
            if num_to_mask > 0:
                masked_indices = np.random.choice(indices, size=num_to_mask, replace=False)
                series[masked_indices] = 0.0
                mask[masked_indices] = 0.0

        return {
            'series': series,
            'series_mask': mask,
            'series_index': row['series_index'],
            'series_index_mask': row['series_index_mask'],
            'series_target': series_target,
            'series_target_mask': series_target_mask,
            'open': row['open'],
            'close': row['close'],
            'open_index': row['open_index'],
            'close_index': row['close_index'],
            'corr_30h': row['corr_30h'],
            'corr_60h': row['corr_60h']
        }