import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import FinancialDataset
from src.model import SimpleReconstructionModel, masked_mse_loss

import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "saved_models/simple_model.pth"

test_dataset = FinancialDataset('data/processed/ml_ready.parquet', '2023-01-01', '2025-12-31', series_mask_fraction=0.7)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = SimpleReconstructionModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

for i, batch in enumerate(test_loader):
    if i >= 5:
        break

    batch = {k: v.to(DEVICE).float() for k, v in batch.items()}
    with torch.no_grad():
        pred = model(batch).squeeze().cpu().numpy()

    target = batch['series_target'].squeeze().cpu().numpy()
    target_mask = batch['series_target_mask'].squeeze().cpu().numpy()
    input_series = batch['series'].squeeze().cpu().numpy()
    input_mask = batch['series_mask'].squeeze().cpu().numpy()
    
    # Skip plotting index_series when reversing normalization
    index_series = batch['series_index'].squeeze().cpu().numpy()
    index_mask = batch['series_index_mask'].squeeze().cpu().numpy()

    index_series = None
    index_mask = None

    x = np.arange(len(target))

    # Reverse low-high normalization (1 - normalized value) * (high - low) + low
    o = batch['open'].squeeze().cpu().numpy()
    try:
        h = batch['high'].squeeze().cpu().numpy()
    except KeyError:
        h = target.max()
    try:
        l = batch['low'].squeeze().cpu().numpy()
    except KeyError:
        l = target.min()

    def reverse_scaling(series, l, h):
        return l + (1 - series) * (h - l)

    pred = reverse_scaling(pred, l, h)
    target = reverse_scaling(target, l, h)
    input_series = reverse_scaling(input_series, l, h)

    plt.figure(figsize=(10, 5))
    plt.plot(x[target_mask == 1], target[target_mask == 1], label="series_target", color='gray', linewidth=2)
    # plt.plot(x[index_mask == 1], index_series[index_mask == 1], label="series_index", color='dimgray', linestyle='dashed', alpha=0.2)
    plt.plot(x, pred, label="series_pred", color='blue', linestyle='solid')
    plt.scatter(x[input_mask == 1], input_series[input_mask == 1], label="series (input)", color='red', s=20, zorder=10)
    plt.legend()
    plt.title(f"Example {i}")
    plt.tight_layout()
    plt.show()
