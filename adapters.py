from abc import ABC, abstractmethod
from pathlib import Path
import sys
import torch
from torch import nn
from safetensors.torch import load_file


class IInputAdapter(nn.Module, ABC):
    """Abstract input adapter interface."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OneHotAdapter(IInputAdapter):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Expected input shape [Batch, Seq, Input_Dim]"
        assert x.size(-1) == self.input_dim, "Unexpected input feature dimension"
        return self.proj(x)


# =====================================================================
#  SpCas9 (crispr-cleavage-predictor) 모델 로드 및 추론 모듈
# =====================================================================

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODEL_REPO_DIR = BASE_DIR / "models" / "crispr-cleavage-predictor"

if str(MODEL_REPO_DIR) not in sys.path:
    sys.path.append(str(MODEL_REPO_DIR))

# backbones.py에서 백본 및 통합 예측 모델 클래스 가져오기
from backbones import CNN_RNN_Backbone, IntegratedPredictor


def load_spcas9_model(weights_path: str = None) -> nn.Module:
    """
    safetensors 가중치(model.safetensors)를 로드하여
    Adapter + Backbone이 결합된 IntegratedPredictor 객체를 반환합니다.
    """
    if weights_path is None:
        weights_path = BASE_DIR / "models" / "SpCas9.safetensors"
        if not weights_path.exists():
            weights_path = BASE_DIR / "models" / "model.safetensors"

    # 1. Adapter 및 Backbone 인스턴스 생성
    adapter = OneHotAdapter(input_dim=4, hidden_dim=128)
    backbone = CNN_RNN_Backbone(input_dim=128)

    # 2. 통합 모델 생성
    model = IntegratedPredictor(adapter=adapter, backbone=backbone)

    # 3. safetensors 가중치 주입
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict)
    model.eval()  # 추론 모드 전환

    return model


def predict_spcas9(one_hot_tensor: torch.Tensor, model: nn.Module = None) -> torch.Tensor:
    """
    UI(app.py) 또는 외부에서 호출하는 추론 인터페이스.
    
    Args:
        one_hot_tensor: [Batch, Seq_Len, 4] 형태의 One-Hot 인코딩 텐서
        model: 미리 로드된 IntegratedPredictor 모델 객체 (없을 시 자동 로드)
    
    Returns:
        torch.Tensor: [Batch, 1] 형태의 예측 절단 효율/활성도 점수
    """
    if model is None:
        model = load_spcas9_model()

    with torch.no_grad():
        # IntegratedPredictor 내부에서 adapter -> backbone 순서로 자동 전달됩니다.
        output = model(one_hot_tensor)

    return output