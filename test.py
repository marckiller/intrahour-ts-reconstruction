import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import FinancialDataset, SimpleReconstructionModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "saved_models/simple_model.pth"

test_dataset = FinancialDataset('data/processed/ml_ready.parquet', '2023-01-01', '2025-12-31', series_mask_fraction=0.8)
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
    index_series = batch['series_index'].squeeze().cpu().numpy()
    index_mask = batch['series_index_mask'].squeeze().cpu().numpy()

    x = np.arange(len(target))

    plt.figure(figsize=(10, 5))
    plt.plot(x[target_mask == 1], target[target_mask == 1], label="series_target", color='gray', linewidth=2)
    plt.plot(x[index_mask == 1], index_series[index_mask == 1], label="series_index", color='dimgray', linestyle='dashed')
    plt.plot(x, pred, label="series_pred", color='lightgray', linestyle='solid')
    plt.scatter(x[input_mask == 1], input_series[input_mask == 1], label="series (input)", color='red', s=20, zorder=10)
    plt.legend()
    plt.title(f"Example {i}")
    plt.tight_layout()
    plt.show()
