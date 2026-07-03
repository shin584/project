import argparse
import torch
import numpy as np

# 모델 조립용 임포트
from adapters import OneHotAdapter
from backbones import CNN_RNN_Backbone, IntegratedPredictor

def seq_to_onehot(seq: str) -> torch.Tensor:
    """단일 DNA 문자열을 파이토치 텐서 [1, Seq_Len, 4] 형태로 변환합니다."""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    # 예외 문자(N 등)는 [0,0,0,0] 처리
    encoded = [mapping.get(nuc.upper(), [0,0,0,0]) for nuc in seq.strip()]
    
    # 텐서 변환 후 Batch 차원(1)을 맨 앞에 추가
    tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
    return tensor

def load_trained_model(weight_path: str) -> IntegratedPredictor:
    """저장된 가중치를 불러와 예측 준비가 완료된 모델을 반환합니다."""
    print(f"Loading model weights from: {weight_path}")
    
    # 1. 모델 뼈대 조립 (훈련 때와 완벽히 동일하게)
    adapter = OneHotAdapter(input_dim=4, hidden_dim=128)
    # 추론 시에는 dropout이 동작하지 않도록 0으로 설정하거나, eval() 모드로 덮어씌웁니다.
    backbone = CNN_RNN_Backbone(input_dim=128, lstm_hidden=128, dropout=0.0)
    model = IntegratedPredictor(adapter=adapter, backbone=backbone)
    
    # 2. 학습된 가중치(뇌) 덮어쓰기
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # 3. 모델을 평가(추론) 모드로 전환 (중요: Dropout, BatchNorm 등 비활성화)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="CRISPR Cleavage Score Predictor")
    parser.add_argument("--sequence", "-s", type=str, required=True, help="예측할 대상 DNA 서열 (예: ATGC...)")
    parser.add_argument("--weight_path", "-w", type=str, required=True, help="학습 완료된 .pth 가중치 파일 경로")
    args = parser.parse_args()

    sequence = args.sequence
    print(f"\nTarget Sequence : {sequence}")
    print(f"Sequence Length : {len(sequence)} bp")

    # 모델 로드 및 텐서 변환
    model = load_trained_model(args.weight_path)
    input_tensor = seq_to_onehot(sequence)

    # 추론 실행 (기울기 계산 비활성화로 메모리 절약 및 속도 향상)
    with torch.no_grad():
        prediction = model.forward(input_tensor)
        
    # 결과 출력
    predicted_score = prediction.item()
    print(f"\nPredicted Cleavage Efficiency Score: {predicted_score:.4f}")

if __name__ == "__main__":
    main()