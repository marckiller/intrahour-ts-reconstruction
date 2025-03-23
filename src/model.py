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
        self.series_hint_proj = nn.Linear(2, hidden_dim)
        
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        ], dim=-1)
        global_embed = self.scalar_proj(scalar_features).unsqueeze(1).expand(-1, S, -1)

        series_and_mask = torch.stack([series, series_mask], dim=-1)  # (B, S, 2)
        series_embed = self.series_hint_proj(series_and_mask)

        x = trend_features + global_embed + series_embed
        x = self.combined_processor(x)
        pred = self.decoder(x).squeeze(-1)
        return pred

def masked_mse_loss(pred, target, mask):
    loss = (pred - target)**2
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)
