# src/training/model.py
import torch
import torch.nn as nn
from typing import Tuple

class ConvLSTMCell(nn.Module):
    def __init__(self , in_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat((x, h_prev), dim=1)
        gates = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur
    
    def init_hidden(self, batch: int, height: int, width: int, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.hidden_channels, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch, self.hidden_channels, height, width, device=device, dtype=dtype)
        return h, c
    

class ConvLSTM(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, return_sequences: bool = True):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=kernel_size)
        self.return_sequences = return_sequences

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        h, c = self.cell.init_hidden(B, H, W, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return h
        

class TrafficConvLSTM(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.convlstm1 = ConvLSTM(in_channels=1, hidden_channels=32, kernel_size=3, return_sequences=True)
        self.convlstm2 = ConvLSTM(in_channels=32, hidden_channels=64, kernel_size=3, return_sequences=False)
        
        self.pool = nn.AdaptiveAvgPool2d((5, 1))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 1, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.act = nn.ReLU()

    def _ensure_btc_hw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("Input must be a 5D tensor.")
        if x.shape[2] != 1 and x.shape[-1] in (1, 3):
            return x.permute(0, 1, 4, 2, 3).contiguous()
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_btc_hw(x)
        x = self.convlstm1(x)
        x = self.convlstm2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out