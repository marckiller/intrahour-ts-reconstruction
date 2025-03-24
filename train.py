import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import FinancialDataset
from src.model import SimpleReconstructionModel, masked_mse_loss


def anchor_loss(pred, target, mask, radius=1):
    loss = 0
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        shifted_mask = torch.roll(mask, shifts=offset, dims=1)
        loss += (((pred - target) ** 2) * shifted_mask).sum()
    return loss / (mask.sum() + 1e-8)

def close_loss(pred, close_val):
    return ((pred[:, -1] - close_val) ** 2).mean()

def smoothness_loss(pred):
    return ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()

DATA_PATH = 'data/processed/ml_ready.parquet'
MODEL_PATH = 'saved_models/simple_model.pth'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = FinancialDataset(DATA_PATH, '2005-01-01', '2017-12-31')
val_dataset = FinancialDataset(DATA_PATH, '2018-01-01', '2020-12-31')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = SimpleReconstructionModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Traning
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = {k: v.to(DEVICE).float() for k, v in batch.items()}
        pred = model(batch)
        mse = masked_mse_loss(pred, batch['series_target'], batch['series_target_mask'])
        anchor = anchor_loss(pred, batch['series_target'], batch['series_mask'])
        close = close_loss(pred, batch['close'])
        smooth = smoothness_loss(pred)
        loss = mse + 0.1 * anchor + 0.1 * close + 0.01 * smooth
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE).float() for k, v in batch.items()}
            pred = model(batch)
            loss = masked_mse_loss(pred, batch['series_target'], batch['series_target_mask'])
            val_loss += loss.item()

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

#Saving
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
