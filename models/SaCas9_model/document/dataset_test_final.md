# 데이터셋 구성에 따른 성능 변화 테스트
* 기존 모델에서 사용된 데이터셋 구성(A375 viability, MOLM13 viabilit)에 293T viability를 추가하여 시도(약 5000여개)

```text
(crispr_env) C:\project\MC_project\SaCas9\crispr-cleavage-predictor>python main.py --mode finetune --data_path dataset/SaCas9_v4.xlsx --weight_path data/spcas9_pretrained_weight.pth --epochs 100
Loading sequence data from: dataset/SaCas9_v4.xlsx
Dataset successfully loaded. Feature shape: torch.Size([5145, 36, 4])
Building One-hot CNN+RNN Backbone...
Total trainable parameters: 141249
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Starting training on local environment...
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0092 - mae: 0.1078 - val_loss: 0.0095 - val_mae: 0.1096 - learning_rate: 0.0001
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0087 - mae: 0.1047 - val_loss: 0.0094 - val_mae: 0.1104 - learning_rate: 0.0001
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0085 - mae: 0.1036 - val_loss: 0.0100 - val_mae: 0.1095 - learning_rate: 0.0001
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0084 - mae: 0.1033 - val_loss: 0.0097 - val_mae: 0.1096 - learning_rate: 0.0001
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0080 - mae: 0.1011 - val_loss: 0.0096 - val_mae: 0.1080 - learning_rate: 0.0001
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0078 - mae: 0.0990 - val_loss: 0.0088 - val_mae: 0.1040 - learning_rate: 0.0001
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0076 - mae: 0.0973 - val_loss: 0.0094 - val_mae: 0.1067 - learning_rate: 0.0001
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0073 - mae: 0.0960 - val_loss: 0.0097 - val_mae: 0.1085 - learning_rate: 0.0001
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0074 - mae: 0.0963 - val_loss: 0.0094 - val_mae: 0.1105 - learning_rate: 0.0001
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0071 - mae: 0.0943 - val_loss: 0.0096 - val_mae: 0.1103 - learning_rate: 0.0001
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0070 - mae: 0.0935 - val_loss: 0.0096 - val_mae: 0.1079 - learning_rate: 0.0001
Learning rate reduced: 0.000100 -> 0.000050
Epoch 12/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0066 - mae: 0.0903 - val_loss: 0.0095 - val_mae: 0.1070 - learning_rate: 0.0001
Epoch 13/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0064 - mae: 0.0894 - val_loss: 0.0093 - val_mae: 0.1083 - learning_rate: 0.0001
Epoch 14/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0064 - mae: 0.0893 - val_loss: 0.0093 - val_mae: 0.1077 - learning_rate: 0.0001
Epoch 15/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0064 - mae: 0.0892 - val_loss: 0.0091 - val_mae: 0.1065 - learning_rate: 0.0001
Epoch 16/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0063 - mae: 0.0877 - val_loss: 0.0096 - val_mae: 0.1090 - learning_rate: 0.0001
Learning rate reduced: 0.000050 -> 0.000025
```

## 추가적인 성능 향상 달성 (val_mae: 0.1260 -> val_mae: 0.1090)
## 최종 데이터 셋 구성 완료
