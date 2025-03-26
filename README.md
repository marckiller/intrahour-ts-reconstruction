# intrahour-ts-reconstruction

This is a work-in-progress project for predicting missing values in high-frequency time series of a financial instrument (like a stock).  
The main idea is to use how the instrument is correlated with a market index to fill in missing data.

We use a machine learning model that runs through a simple API built with FastAPI.

## What it does

- Takes a time series of 60 minutes (some values may be missing)
- Uses a full 60-minute time series of a correlated index
- Uses basic price data (open, high, low, close) for both the instrument and the index
- Uses correlation values (30-hour and 60-hour)
- Returns a completed 60-element time series for the instrument

## How to run

1. Install the required packages
2. Start the API with:

```bash
uvicorn app:app --reload
```

3.	Go to http://localhost:8000/docs to test the API in your browser
4.	Enter sample JSON
   
```json
{
  "series": [null, null, null, null, null, null, 7.308, 7.328, 7.35, null, null, null, null, null, null, 7.35, 7.34, null, 7.356, null, 7.374, null, null, 7.412, null, null, null, null, null, null, null, null, 7.398, null, null, 7.398, null, 7.376, null, null, null, null, null, null, 7.392, null, null, 7.39, 7.398, 7.4, null, 7.416, null, null, null, null, 7.396, null, null, null],
  "series_index": [2371.01, 2371.41, 2372.45, 2365.8, 2370.7, 2371.31, 2372.02, 2368.01, 2369.54, 2371.39, 2370.62, 2369.84, 2373.18, 2373.03, 2374.52, 2374.97, 2374.2, 2373.71, 2373.45, 2373.85, 2376.17, 2376.5, 2378.7, 2379.3, 2376.64, 2378.83, 2380.24, 2380.58, 2380.83, 2380.91, 2381.49, 2380.2, 2380.86, 2380.84, 2380.42, 2379.8, 2379.18, 2379.85, 2379.53, 2379.19, 2377.83, 2377.85, 2376.51, 2375.26, 2376.7, 2375.97, 2375.45, 2375.72, 2375.6, 2375.89, 2375.8, 2374.89, 2373.98, 2374.03, 2374.02, 2374.71, 2374.27, 2374.45, 2374.01, 2373.91],
  "open": 7.254,
  "high": 7.43,
  "low": 7.254,
  "close": 7.4,
  "open_index": 2369.06,
  "high_index": 2381.87,
  "low_index": 2364.82,
  "close_index": 2373.91,
  "corr_30h": 0.6544904481137022,
  "corr_60h": 0.5951298016627088
}
```
# Status

This is a development version. Things may change. Especially models used.
