from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import pandas as pd
from functions.api_processing import *
from src.model import SimpleReconstructionModel

app = FastAPI()

model = SimpleReconstructionModel()
model.load_state_dict(torch.load("saved_models/simple_model.pth", map_location=torch.device("cpu")))
model.eval()

class InputData(BaseModel):
    series: List[Optional[float]]
    series_index: List[Optional[float]]
    open: float
    high: float
    low: float
    close: float
    open_index: float
    high_index: float
    low_index: float
    close_index: float
    corr_30h: float
    corr_60h: float

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])

    if len(df['series'][0]) != 60 or len(df['series_index'][0]) != 60:
        return {"status": "error", "message": "series and series_index must be of length 60"}
    
    df['series_mask'] = df['series'].apply(lambda x: np.array([0.0 if v is None else 1.0 for v in x], dtype=np.float32))
    df['series_index_mask'] = df['series_index'].apply(lambda x: np.array([0.0 if v is None else 1.0 for v in x], dtype=np.float32))
    
    df['series'] = df['series'].apply(lambda x: np.array([0.0 if v is None else float(v) for v in x], dtype=np.float32))
    df['series_index'] = df['series_index'].apply(lambda x: np.array([0.0 if v is None else float(v) for v in x], dtype=np.float32))
    
    df = normalize_scalar_column(df, 'open', 'low', 'high')
    df = normalize_scalar_column(df, 'close', 'low', 'high')
    df = normalize_scalar_column(df, 'open_index', 'low_index', 'high_index')
    df = normalize_scalar_column(df, 'close_index', 'low_index', 'high_index')

    df = normalize_series_column(df, 'series', 'series_mask', 'low', 'high')
    df = normalize_series_column(df, 'series_index', 'series_index_mask', 'low_index', 'high_index')
    
    with torch.no_grad():
        input_tensor = {
            'series': torch.tensor(np.stack(df['series'].values), dtype=torch.float32),
            'series_mask': torch.tensor(np.stack(df['series_mask'].values), dtype=torch.float32),
            'series_index': torch.tensor(np.stack(df['series_index'].values), dtype=torch.float32),
            'series_index_mask': torch.tensor(np.stack(df['series_index_mask'].values), dtype=torch.float32),
            'open': torch.tensor(df['open'].values, dtype=torch.float32),
            'close': torch.tensor(df['close'].values, dtype=torch.float32),
            'open_index': torch.tensor(df['open_index'].values, dtype=torch.float32),
            'close_index': torch.tensor(df['close_index'].values, dtype=torch.float32),
            'corr_30h': torch.tensor(df['corr_30h'].values, dtype=torch.float32),
            'corr_60h': torch.tensor(df['corr_60h'].values, dtype=torch.float32),
        }

        output_tensor = model(input_tensor)
        output_np = output_tensor.numpy()[0]
        denorm_output = denormalize_series( output_np, df['low'].values[0], df['high'].values[0])

        output = denorm_output.tolist()

    return {"status": "ok", "prediction": output}

@app.get("/")
def read_root():
    return {"Hello": "World"}