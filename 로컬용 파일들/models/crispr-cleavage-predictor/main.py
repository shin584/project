import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold

# 어댑터, 백본, 파이프라인 임포트
from adapters import OneHotAdapter
from backbones import CNN_RNN_Backbone, IntegratedPredictor
from pipeline.pipeline_manager import PipelineManager

# 훈련 편의를 위한 콜백 및 어댑터 임포트 (화면 출력 및 조기 종료)
from training.adapters import PyTorchLossAdapter, PyTorchOptimizerAdapter
from training.callbacks import EarlyStoppingCallback, LRSchedulerCallback, ProgressBarLogger

def _load_tensor_dataset(data_path: str) -> TensorDataset:
    print(f"Loading sequence data from: {data_path}")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path, engine='openpyxl')
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file format. Please provide .csv or .xlsx")

    seq_col = next((c for c in df.columns if c.lower() in ["input_sequence", "sequence", "target_sequence", "seq"]), df.columns[0])
    label_col = next((c for c in df.columns if c.lower() in ["score", "label", "efficiency", "cleavage_efficiency", "cleavage_score"]), df.columns[1])
    
    sequences = df[seq_col].astype(str).tolist()
    labels = df[label_col].astype(float).tolist()
    
    def seq_to_onehot(seq: str) -> torch.Tensor:
        mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
        encoded = [mapping.get(nuc.upper(), [0,0,0,0]) for nuc in seq]
        return torch.tensor(encoded, dtype=torch.float32)

    feature_tensor = torch.stack([seq_to_onehot(seq) for seq in sequences], dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    if feature_tensor.shape[-1] != 4:
        raise ValueError(f"Expected one-hot channel size 4, got {feature_tensor.shape[-1]}")
    print(f"Dataset successfully loaded. Feature shape: {feature_tensor.shape}")
    return TensorDataset(feature_tensor, label_tensor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "finetune"])
    parser.add_argument("--weight_path", type=str, default=None)
    parser.add_argument("--dropout_rate", type=float, default=0.4, help="Dropout probability (0~1)")
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"], help="Loss function type")
    args = parser.parse_args()

    # 1. 데이터 로드 및 분할
    dataset = _load_tensor_dataset(args.data_path)
    dataset_size = len(dataset)
    test_size = max(1, int(dataset_size * 0.1))
    train_val_size = dataset_size - test_size
    train_val_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. 최신 모델 조립
    if args.mode == "finetune":
        if not args.weight_path:
            raise ValueError("--weight_path is required when --mode is 'finetune'.")

    print("Building One-hot CNN+RNN Backbone...")
    dropout_rate = args.dropout_rate

    def build_model() -> IntegratedPredictor:
        adapter = OneHotAdapter(input_dim=4, hidden_dim=128)
        backbone = CNN_RNN_Backbone(input_dim=128, lstm_hidden=128, dropout=dropout_rate)
        model = IntegratedPredictor(adapter=adapter, backbone=backbone)
        if args.mode == "finetune":
            print(f"Loading pretrained weights from: {args.weight_path}")
            state_dict = torch.load(args.weight_path, map_location="cpu")
            model.load_state_dict(state_dict)
        return model

    param_probe = build_model()
    print(
        f"Total trainable parameters: {sum(p.numel() for p in param_probe.parameters() if p.requires_grad)}"
    )

    # 3. 옵티마이저, 스케줄러 및 콜백 세팅 (코랩과 동일하게!)
    lr = 1e-4 if args.mode == "finetune" else 1e-3
    criterion_map = {"mse": torch.nn.MSELoss, "huber": torch.nn.HuberLoss}
    
    # 로컬에 모델 가중치를 저장할 폴더 생성
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)

    def loss_factory() -> PyTorchLossAdapter:
        return PyTorchLossAdapter(criterion_map[args.loss]())

    def optimizer_factory(model: IntegratedPredictor):
        raw_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        return raw_optimizer, PyTorchOptimizerAdapter(raw_optimizer)

    def scheduler_factory(raw_optimizer: torch.optim.Optimizer):
        return ReduceLROnPlateau(raw_optimizer, mode="min", factor=0.5, patience=4)

    def callbacks_factory(model: IntegratedPredictor, scheduler: object | None, fold_index: int):
        callbacks = [
            ProgressBarLogger(total_epochs=args.epochs),
            EarlyStoppingCallback(
                model=model,
                patience=10,
                save_path=os.path.join(save_dir, f"best_model_fold{fold_index}.pth"),
            ),
        ]
        if scheduler is not None:
            callbacks.append(LRSchedulerCallback(scheduler))
        return callbacks

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_loaders = []
    for train_indices, val_indices in kfold.split(range(len(train_val_set))):
        train_subset = Subset(train_val_set, train_indices)
        val_subset = Subset(train_val_set, val_indices)
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False
        )
        fold_loaders.append((train_loader, val_loader))

    manager = PipelineManager(
        model_factory=build_model,
        loss_factory=loss_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        callbacks_factory=callbacks_factory,
    )

    print("Starting K-Fold training on local environment...")
    fold_val_losses = manager.run_train_pipeline(epochs=args.epochs, fold_loaders=fold_loaders)

    if fold_val_losses:
        best_fold_index = min(range(len(fold_val_losses)), key=fold_val_losses.__getitem__) + 1
        best_model_path = os.path.join(save_dir, f"best_model_fold{best_fold_index}.pth")

        if os.path.exists(best_model_path):
            test_model = build_model()
            state_dict = torch.load(best_model_path, map_location="cpu")
            test_model.load_state_dict(state_dict)

            test_criterion = criterion_map[args.loss]()
            test_model.eval()
            total_loss = 0.0
            total_mae = 0.0
            batch_count = 0

            with torch.no_grad():
                for features, labels in test_loader:
                    predictions = test_model.forward(features)
                    loss = test_criterion(predictions, labels)
                    mae = F.l1_loss(predictions, labels, reduction="mean")
                    total_loss += float(loss.item()) if hasattr(loss, "item") else float(loss)
                    total_mae += float(mae.item())
                    batch_count += 1

            avg_test_loss = total_loss / batch_count if batch_count > 0 else 0.0
            avg_test_mae = total_mae / batch_count if batch_count > 0 else 0.0
            print(
                "Final Test metrics "
                f"(loss={avg_test_loss:.4f}, mae={avg_test_mae:.4f})"
            )
        else:
            print(f"Best model checkpoint not found: {best_model_path}")

if __name__ == "__main__":
    main()