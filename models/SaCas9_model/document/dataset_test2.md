# 데이터셋 구성에 따른 성능 변화 테스트

* SaCas9의 원본 데이터셋(약 7000여개)를 모두 사용

```text
(crispr_env) C:\project\MC_project\SaCas9\crispr-cleavage-predictor>python main.py --mode finetune --data_path "dataset/Supplementary_Table_1_Saureus_model_input.xlsx" --epochs 100 --batch_size 64 --weight_path "data/spcas9_pretrained_weight.pth"
Loading sequence data from: dataset/Supplementary_Table_1_Saureus_model_input.xlsx
Dataset successfully loaded. Feature shape: torch.Size([7465, 36, 4])
Building One-hot CNN+RNN Backbone...
Total trainable parameters: 141249
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Starting training on local environment...
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0782 - mae: 0.2243 - val_loss: 0.0792 - val_mae: 0.2286 - learning_rate: 0.0001
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0727 - mae: 0.2188 - val_loss: 0.0790 - val_mae: 0.2304 - learning_rate: 0.0001
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0700 - mae: 0.2144 - val_loss: 0.0767 - val_mae: 0.2252 - learning_rate: 0.0001
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0685 - mae: 0.2130 - val_loss: 0.0756 - val_mae: 0.2245 - learning_rate: 0.0001
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0658 - mae: 0.2078 - val_loss: 0.0754 - val_mae: 0.2257 - learning_rate: 0.0001
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0649 - mae: 0.2072 - val_loss: 0.0756 - val_mae: 0.2249 - learning_rate: 0.0001
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0630 - mae: 0.2038 - val_loss: 0.0736 - val_mae: 0.2221 - learning_rate: 0.0001
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0614 - mae: 0.2004 - val_loss: 0.0759 - val_mae: 0.2239 - learning_rate: 0.0001
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0604 - mae: 0.1983 - val_loss: 0.0725 - val_mae: 0.2203 - learning_rate: 0.0001
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0590 - mae: 0.1960 - val_loss: 0.0737 - val_mae: 0.2218 - learning_rate: 0.0001
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0580 - mae: 0.1943 - val_loss: 0.0727 - val_mae: 0.2200 - learning_rate: 0.0001
Epoch 12/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0568 - mae: 0.1921 - val_loss: 0.0725 - val_mae: 0.2183 - learning_rate: 0.0001
Epoch 13/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0555 - mae: 0.1898 - val_loss: 0.0740 - val_mae: 0.2221 - learning_rate: 0.0001
Epoch 14/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0551 - mae: 0.1889 - val_loss: 0.0713 - val_mae: 0.2158 - learning_rate: 0.0001
Epoch 15/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0538 - mae: 0.1863 - val_loss: 0.0719 - val_mae: 0.2155 - learning_rate: 0.0001
Epoch 16/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0524 - mae: 0.1834 - val_loss: 0.0717 - val_mae: 0.2159 - learning_rate: 0.0001
Epoch 17/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0519 - mae: 0.1822 - val_loss: 0.0722 - val_mae: 0.2133 - learning_rate: 0.0001
Epoch 18/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0508 - mae: 0.1801 - val_loss: 0.0708 - val_mae: 0.2145 - learning_rate: 0.0001
Epoch 19/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0494 - mae: 0.1774 - val_loss: 0.0699 - val_mae: 0.2119 - learning_rate: 0.0001
Epoch 20/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0486 - mae: 0.1754 - val_loss: 0.0698 - val_mae: 0.2132 - learning_rate: 0.0001
Epoch 21/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0485 - mae: 0.1752 - val_loss: 0.0703 - val_mae: 0.2111 - learning_rate: 0.0001
Epoch 22/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0472 - mae: 0.1729 - val_loss: 0.0702 - val_mae: 0.2099 - learning_rate: 0.0001
Epoch 23/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0460 - mae: 0.1701 - val_loss: 0.0718 - val_mae: 0.2121 - learning_rate: 0.0001
Epoch 24/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0452 - mae: 0.1681 - val_loss: 0.0701 - val_mae: 0.2093 - learning_rate: 0.0001
Epoch 25/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0446 - mae: 0.1670 - val_loss: 0.0704 - val_mae: 0.2099 - learning_rate: 0.0001
Learning rate reduced: 0.000100 -> 0.000050
Epoch 26/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0431 - mae: 0.1641 - val_loss: 0.0704 - val_mae: 0.2080 - learning_rate: 0.0001
Epoch 27/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0423 - mae: 0.1617 - val_loss: 0.0695 - val_mae: 0.2078 - learning_rate: 0.0001
Epoch 28/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0419 - mae: 0.1612 - val_loss: 0.0712 - val_mae: 0.2081 - learning_rate: 0.0001
Epoch 29/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0415 - mae: 0.1601 - val_loss: 0.0704 - val_mae: 0.2082 - learning_rate: 0.0001
Epoch 30/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0412 - mae: 0.1598 - val_loss: 0.0702 - val_mae: 0.2068 - learning_rate: 0.0001
Epoch 31/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0404 - mae: 0.1580 - val_loss: 0.0699 - val_mae: 0.2065 - learning_rate: 0.0001
Epoch 32/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0402 - mae: 0.1575 - val_loss: 0.0715 - val_mae: 0.2084 - learning_rate: 0.0001
Learning rate reduced: 0.000050 -> 0.000025
Epoch 33/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0393 - mae: 0.1554 - val_loss: 0.0700 - val_mae: 0.2059 - learning_rate: 0.0000
Epoch 34/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0391 - mae: 0.1550 - val_loss: 0.0696 - val_mae: 0.2056 - learning_rate: 0.0000
Epoch 35/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0389 - mae: 0.1545 - val_loss: 0.0697 - val_mae: 0.2056 - learning_rate: 0.0000
Epoch 36/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0388 - mae: 0.1541 - val_loss: 0.0701 - val_mae: 0.2060 - learning_rate: 0.0000
Epoch 37/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0384 - mae: 0.1536 - val_loss: 0.0697 - val_mae: 0.2052 - learning_rate: 0.0000
Learning rate reduced: 0.000025 -> 0.000013
```

* 성능이 오히려 하락함(val_mae: 0.1812 -> val_mae: 0.2052)
