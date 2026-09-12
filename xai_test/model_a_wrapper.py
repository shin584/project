import numpy as np
import torch
from torch import nn


class OneHotAdapter(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        # 가중치 형태 [128, 4]에 맞게 nn.Linear로 변경
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x shape: [Batch, 36, 4]
        x = self.proj(x)  # -> [Batch, 36, 128]
        return x.transpose(1, 2)  # CNN 백본 입력에 맞게 변경 -> [Batch, 128, 36]


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
        self.conv = nn.Conv1d(
            input_dim, conv_channels, kernel_size, padding=kernel_size // 2
        )
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

    def forward(self, x):
        # x shape: [Batch, 128, 36]
        x = self.act(self.conv(x))
        x = x.transpose(1, 2)  # [Batch, 36, 64]
        out, _ = self.lstm(x)  # out shape: [Batch, 36, 128]

        # 마지막 시점(sequence)의 hidden state를 추출하여 분류기 통과
        out = self.head(out[:, -1, :])
        return out


class IntegratedPredictor(nn.Module):
    def __init__(self, adapter: nn.Module, backbone: nn.Module):
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone

    def forward(self, x):
        x = self.adapter(x)
        x = self.backbone(x)
        return x.squeeze(-1)  # 1D Array 변환


class Model_A_Predictor:
    def __init__(self, weight_path: str | None = None, dropout_rate: float = 0.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 원본 구조와 동일하게 어댑터와 백본 조립
        adapter = OneHotAdapter(input_dim=4, hidden_dim=128)
        backbone = CNN_RNN_Backbone(
            input_dim=128, lstm_hidden=128, dropout=dropout_rate
        )
        self.model = IntegratedPredictor(adapter=adapter, backbone=backbone).to(
            self.device
        )

        # 가중치 로드
        if weight_path:
            self.model.load_state_dict(
                torch.load(weight_path, map_location=self.device)
            )
        self.model.eval()

        self.nuc_map = {
            "A": [1.0, 0.0, 0.0, 0.0],
            "C": [0.0, 1.0, 0.0, 0.0],
            "G": [0.0, 0.0, 1.0, 0.0],
            "T": [0.0, 0.0, 0.0, 1.0],
        }

    def encode_one_hot(self, sequences: list[str]) -> torch.Tensor:
        encoded_list = []
        for seq in sequences:
            encoded = [self.nuc_map.get(n.upper(), [0.0, 0.0, 0.0, 0.0]) for n in seq]
            encoded_list.append(encoded)
        tensor = torch.tensor(encoded_list, dtype=torch.float32)
        return tensor  # shape: [Batch, 36, 4]

    def predict(self, sequences: list[str]) -> np.ndarray:
        x = self.encode_one_hot(sequences).to(self.device)
        with torch.no_grad():
            preds = self.model(x).cpu().numpy()
        return preds
