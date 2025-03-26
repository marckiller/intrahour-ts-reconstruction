import numpy as np
import pandas as pd

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

def denormalize_series(normalized_series: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Denormalize a normalized series using the original low and high values.
    Mask should indicate which values were originally present (1.0) or missing (0.0).
    """
    denorm = normalized_series * (high - low) + low
    return denorm