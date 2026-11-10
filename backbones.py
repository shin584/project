import torch
from torch import nn


class CNN_RNN_Backbone(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        conv_channels: int = 64,
        kernel_size: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.conv = nn.Conv1d(input_dim, conv_channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Expected input shape [Batch, Seq, 128]"
        assert x.size(-1) == self.input_dim, "Unexpected input feature dimension"
        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


class IntegratedPredictor(nn.Module):
    def __init__(self, adapter: nn.Module, backbone: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return self.backbone(x)
