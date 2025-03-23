import torch
import torch.nn as nn
import os

os.makedirs("saved_models", exist_ok=True)

class SimpleReconstructionModel(nn.Module):
    def __init__(self, seq_len=60, hidden_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.series_index_proj = nn.Linear(1, hidden_dim)
        self.scalar_proj = nn.Linear(6, hidden_dim)
        self.index_conv = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch):
        series = batch['series']
        series_mask = batch['series_mask']
        series_index = batch['series_index']
        series_index_mask = batch['series_index_mask']

        B, S = series.shape

        trend = torch.diff(series_index, dim=1, prepend=series_index[:, :1])
        trend_conv_input = trend.unsqueeze(1)
        trend_features = self.index_conv(trend_conv_input).transpose(1, 2)

        scalar_features = torch.stack([
            batch['open'], batch['close'],
            batch['open_index'], batch['close_index'],
            batch['corr_30h'], batch['corr_60h']
        ], dim=-1)  # (B, 6)
        scalar_embed = self.scalar_proj(scalar_features).unsqueeze(1).expand(-1, S, -1)

        x = torch.cat([trend_features, scalar_embed], dim=-1)
        x = self.combined_processor(x)
        pred = self.decoder(x).squeeze(-1)
        pred = torch.where(series_mask == 1, series, 0.7 * series + 0.3 * pred)
        pred = torch.where(series_mask == 1, series, pred)
        return pred

def masked_mse_loss(pred, target, mask):
    loss = (pred - target)**2
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)
