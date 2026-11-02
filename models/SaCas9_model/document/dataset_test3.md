# 데이터셋 구성에 따른 성능 변화 테스트
* 기존 모델에서 사용된 데이터셋 구성(A375 viability, MOLM13 viabilit)을 사용하여 다시 시도(약 3600여개)
* 기준 모델 데이터셋 구성 + 파인튜닝

```text
(crispr_env) C:\project\MC_project\SaCas9\crispr-cleavage-predictor>python main.py --mode finetune --data_path "dataset/Supplementary_Table_1_Saureus_model_input_v3.xlsx" --epochs 100 --batch_size 64 --weight_path "data/spcas9_pretrained_weight.pth"
Loading sequence data from: dataset/Supplementary_Table_1_Saureus_model_input_v3.xlsx
Dataset successfully loaded. Feature shape: torch.Size([3610, 36, 4])
Building One-hot CNN+RNN Backbone...
Total trainable parameters: 141249
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Starting training on local environment...
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0246 - mae: 0.1254 - val_loss: 0.0225 - val_mae: 0.1231 - learning_rate: 0.0001
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0226 - mae: 0.1205 - val_loss: 0.0226 - val_mae: 0.1235 - learning_rate: 0.0001
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0212 - mae: 0.1167 - val_loss: 0.0226 - val_mae: 0.1239 - learning_rate: 0.0001
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0204 - mae: 0.1138 - val_loss: 0.0235 - val_mae: 0.1258 - learning_rate: 0.0001
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0196 - mae: 0.1114 - val_loss: 0.0237 - val_mae: 0.1265 - learning_rate: 0.0001
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0193 - mae: 0.1107 - val_loss: 0.0242 - val_mae: 0.1279 - learning_rate: 0.0001
Learning rate reduced: 0.000100 -> 0.000050
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0181 - mae: 0.1074 - val_loss: 0.0230 - val_mae: 0.1246 - learning_rate: 0.0001
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0177 - mae: 0.1060 - val_loss: 0.0221 - val_mae: 0.1229 - learning_rate: 0.0001
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0174 - mae: 0.1051 - val_loss: 0.0230 - val_mae: 0.1247 - learning_rate: 0.0001
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0173 - mae: 0.1046 - val_loss: 0.0228 - val_mae: 0.1240 - learning_rate: 0.0001
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0168 - mae: 0.1030 - val_loss: 0.0234 - val_mae: 0.1252 - learning_rate: 0.0001
Epoch 12/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0167 - mae: 0.1029 - val_loss: 0.0229 - val_mae: 0.1240 - learning_rate: 0.0001
Epoch 13/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0163 - mae: 0.1012 - val_loss: 0.0233 - val_mae: 0.1250 - learning_rate: 0.0001
Learning rate reduced: 0.000050 -> 0.000025
Epoch 14/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0160 - mae: 0.1001 - val_loss: 0.0234 - val_mae: 0.1256 - learning_rate: 0.0000
Epoch 15/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0160 - mae: 0.1001 - val_loss: 0.0227 - val_mae: 0.1239 - learning_rate: 0.0000
Epoch 16/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0157 - mae: 0.0994 - val_loss: 0.0228 - val_mae: 0.1243 - learning_rate: 0.0000
Epoch 17/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0155 - mae: 0.0989 - val_loss: 0.0230 - val_mae: 0.1243 - learning_rate: 0.0000
Epoch 18/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0154 - mae: 0.0985 - val_loss: 0.0238 - val_mae: 0.1260 - learning_rate: 0.0000
Learning rate reduced: 0.000025 -> 0.000013
```

## 기준 성능 초과 달성 (val_mae: 0.1342 -> val_mae: 0.1260)
