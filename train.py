import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import FinancialDataset, masked_mse_loss
import os

class SimpleReconstructionModel(nn.Module):
    def __init__(self, hidden_dim=64):
        super(SimpleReconstructionModel, self).__init__()
        self.index_conv = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, batch):
        series_index = batch['series_index']
        series_mask = batch['series_mask']
        scalar_embed = batch['scalar_embed']

        trend = torch.diff(series_index, dim=1, prepend=series_index[:, :1])
        trend_conv_input = trend.unsqueeze(1)
        trend_features = self.index_conv(trend_conv_input).transpose(1, 2)

        x = torch.cat([trend_features, scalar_embed], dim=-1)
        pred = self.decoder(x).squeeze(-1)
        pred = torch.where(series_mask == 1, series, 0.7 * series + 0.3 * pred)
        pred = torch.where(series_mask == 1, series, pred)
        
        return pred

DATA_PATH = 'data/processed/ml_ready.parquet'
MODEL_PATH = 'saved_models/simple_model.pth'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
BATCH_SIZE = 32
EPOCHS = 10
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
        loss = masked_mse_loss(pred, batch['series_target'], batch['series_target_mask'])
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
