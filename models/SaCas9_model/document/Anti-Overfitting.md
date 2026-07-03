# 조기 과적합 현상을 잡기 위해 모델 고도화 작업 수행
* dropout rate를 0.4로 높임
* Loss 함수를 `MSELoss`에서 `HuberLoss`로 교체
* K-Fold Cross Validation 도입
* test set 분할 및 평가 기능 도입

```text
(crispr_env) C:\project\MC_project\SaCas9\crispr-cleavage-predictor>python main.py --mode finetune --data_path dataset/SaCas9_v4.xlsx --weight_path data/spcas9_pretrained_weight.pth --epochs 100
Loading sequence data from: dataset/SaCas9_v4.xlsx
Dataset successfully loaded. Feature shape: torch.Size([5145, 36, 4])
Building One-hot CNN+RNN Backbone...
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Total trainable parameters: 141249
Starting K-Fold training on local environment...
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0078 - mae: 0.0987 - val_loss: 0.0083 - val_mae: 0.1017 - learning_rate: 0.0001
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0076 - mae: 0.0975 - val_loss: 0.0085 - val_mae: 0.1029 - learning_rate: 0.0001
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0072 - mae: 0.0952 - val_loss: 0.0086 - val_mae: 0.1032 - learning_rate: 0.0001
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0069 - mae: 0.0929 - val_loss: 0.0087 - val_mae: 0.1041 - learning_rate: 0.0001
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0068 - mae: 0.0923 - val_loss: 0.0089 - val_mae: 0.1054 - learning_rate: 0.0001
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0066 - mae: 0.0910 - val_loss: 0.0091 - val_mae: 0.1068 - learning_rate: 0.0001
Learning rate reduced: 0.000100 -> 0.000050
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0064 - mae: 0.0888 - val_loss: 0.0090 - val_mae: 0.1059 - learning_rate: 0.0001
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0063 - mae: 0.0877 - val_loss: 0.0091 - val_mae: 0.1067 - learning_rate: 0.0001
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0062 - mae: 0.0872 - val_loss: 0.0094 - val_mae: 0.1073 - learning_rate: 0.0001
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0061 - mae: 0.0867 - val_loss: 0.0091 - val_mae: 0.1052 - learning_rate: 0.0001
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0062 - mae: 0.0874 - val_loss: 0.0092 - val_mae: 0.1061 - learning_rate: 0.0001
Learning rate reduced: 0.000050 -> 0.000025
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0080 - mae: 0.1000 - val_loss: 0.0075 - val_mae: 0.0965 - learning_rate: 0.0001
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0075 - mae: 0.0966 - val_loss: 0.0078 - val_mae: 0.0985 - learning_rate: 0.0001
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0073 - mae: 0.0953 - val_loss: 0.0079 - val_mae: 0.0985 - learning_rate: 0.0001
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0071 - mae: 0.0939 - val_loss: 0.0080 - val_mae: 0.0992 - learning_rate: 0.0001
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0068 - mae: 0.0921 - val_loss: 0.0084 - val_mae: 0.1022 - learning_rate: 0.0001
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0068 - mae: 0.0925 - val_loss: 0.0088 - val_mae: 0.1044 - learning_rate: 0.0001
Learning rate reduced: 0.000100 -> 0.000050
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0064 - mae: 0.0889 - val_loss: 0.0089 - val_mae: 0.1057 - learning_rate: 0.0001
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0063 - mae: 0.0883 - val_loss: 0.0081 - val_mae: 0.1004 - learning_rate: 0.0001
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0062 - mae: 0.0877 - val_loss: 0.0082 - val_mae: 0.1004 - learning_rate: 0.0001
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0061 - mae: 0.0865 - val_loss: 0.0081 - val_mae: 0.1000 - learning_rate: 0.0001
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0061 - mae: 0.0867 - val_loss: 0.0086 - val_mae: 0.1020 - learning_rate: 0.0001
Learning rate reduced: 0.000050 -> 0.000025
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0078 - mae: 0.0991 - val_loss: 0.0082 - val_mae: 0.1020 - learning_rate: 0.0001
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0074 - mae: 0.0968 - val_loss: 0.0083 - val_mae: 0.1025 - learning_rate: 0.0001
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0073 - mae: 0.0959 - val_loss: 0.0088 - val_mae: 0.1056 - learning_rate: 0.0001
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0070 - mae: 0.0930 - val_loss: 0.0095 - val_mae: 0.1097 - learning_rate: 0.0001
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0071 - mae: 0.0945 - val_loss: 0.0087 - val_mae: 0.1049 - learning_rate: 0.0001
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0067 - mae: 0.0913 - val_loss: 0.0090 - val_mae: 0.1067 - learning_rate: 0.0001
Learning rate reduced: 0.000100 -> 0.000050
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0063 - mae: 0.0882 - val_loss: 0.0092 - val_mae: 0.1072 - learning_rate: 0.0001
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0062 - mae: 0.0873 - val_loss: 0.0089 - val_mae: 0.1059 - learning_rate: 0.0001
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0061 - mae: 0.0866 - val_loss: 0.0092 - val_mae: 0.1072 - learning_rate: 0.0001
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0061 - mae: 0.0866 - val_loss: 0.0090 - val_mae: 0.1059 - learning_rate: 0.0001
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0060 - mae: 0.0856 - val_loss: 0.0089 - val_mae: 0.1052 - learning_rate: 0.0001
Learning rate reduced: 0.000050 -> 0.000025
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0078 - mae: 0.0991 - val_loss: 0.0077 - val_mae: 0.0978 - learning_rate: 0.0001
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0077 - mae: 0.0981 - val_loss: 0.0084 - val_mae: 0.1020 - learning_rate: 0.0001
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0073 - mae: 0.0959 - val_loss: 0.0083 - val_mae: 0.1019 - learning_rate: 0.0001
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0070 - mae: 0.0941 - val_loss: 0.0085 - val_mae: 0.1027 - learning_rate: 0.0001
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0070 - mae: 0.0936 - val_loss: 0.0082 - val_mae: 0.1005 - learning_rate: 0.0001
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0068 - mae: 0.0926 - val_loss: 0.0084 - val_mae: 0.1030 - learning_rate: 0.0001
Learning rate reduced: 0.000100 -> 0.000050
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0064 - mae: 0.0897 - val_loss: 0.0083 - val_mae: 0.1018 - learning_rate: 0.0001
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0063 - mae: 0.0887 - val_loss: 0.0084 - val_mae: 0.1015 - learning_rate: 0.0001
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0062 - mae: 0.0883 - val_loss: 0.0083 - val_mae: 0.1008 - learning_rate: 0.0001
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0062 - mae: 0.0877 - val_loss: 0.0086 - val_mae: 0.1031 - learning_rate: 0.0001
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0061 - mae: 0.0868 - val_loss: 0.0085 - val_mae: 0.1023 - learning_rate: 0.0001
Learning rate reduced: 0.000050 -> 0.000025
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0080 - mae: 0.1002 - val_loss: 0.0078 - val_mae: 0.1004 - learning_rate: 0.0001
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0076 - mae: 0.0976 - val_loss: 0.0078 - val_mae: 0.0990 - learning_rate: 0.0001
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0074 - mae: 0.0955 - val_loss: 0.0082 - val_mae: 0.1025 - learning_rate: 0.0001
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0073 - mae: 0.0959 - val_loss: 0.0082 - val_mae: 0.1016 - learning_rate: 0.0001
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0070 - mae: 0.0930 - val_loss: 0.0080 - val_mae: 0.1002 - learning_rate: 0.0001
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0068 - mae: 0.0917 - val_loss: 0.0083 - val_mae: 0.1025 - learning_rate: 0.0001
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0068 - mae: 0.0916 - val_loss: 0.0082 - val_mae: 0.1016 - learning_rate: 0.0001
Learning rate reduced: 0.000100 -> 0.000050
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0063 - mae: 0.0877 - val_loss: 0.0081 - val_mae: 0.1012 - learning_rate: 0.0001
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0063 - mae: 0.0877 - val_loss: 0.0081 - val_mae: 0.1009 - learning_rate: 0.0001
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0061 - mae: 0.0861 - val_loss: 0.0081 - val_mae: 0.1010 - learning_rate: 0.0001
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0061 - mae: 0.0863 - val_loss: 0.0082 - val_mae: 0.1015 - learning_rate: 0.0001
Epoch 12/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0060 - mae: 0.0853 - val_loss: 0.0084 - val_mae: 0.1031 - learning_rate: 0.0001
Learning rate reduced: 0.000050 -> 0.000025
K-Fold average validation metrics (loss=0.0087, mae=0.1037)
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Final Test metrics (loss=0.0109, mae=0.1197)
```
