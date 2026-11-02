# SpCas9 데이터셋을 이용한 프리트레이닝 결과
* 새로 구축한 모델에 데이터가 풍부한 SpCas9 데이터셋으로 사전학습 시행 후 가중치 추출 

```text
Loading sequence data from: dataset/SpCas9_pretraing.xlsx
Dataset successfully loaded. Feature shape: torch.Size([36445, 30, 4])
Building One-hot CNN+RNN Backbone...
Total trainable parameters: 141249
Starting training on local environment...
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 479.5404 - mae: 19.1663 - val_loss: 413.9416 - val_mae: 17.7602 - learning_rate: 0.0010
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 266.5318 - mae: 12.7961 - val_loss: 98.3445 - val_mae: 6.8310 - learning_rate: 0.0010
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 69.9732 - mae: 5.8148 - val_loss: 63.8867 - val_mae: 5.5653 - learning_rate: 0.0010
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 58.0711 - mae: 5.2886 - val_loss: 62.1675 - val_mae: 5.5488 - learning_rate: 0.0010
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 52.3410 - mae: 4.9885 - val_loss: 55.4706 - val_mae: 5.0984 - learning_rate: 0.0010
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 49.9303 - mae: 4.8450 - val_loss: 49.0537 - val_mae: 4.8035 - learning_rate: 0.0010
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 47.7463 - mae: 4.7308 - val_loss: 55.7966 - val_mae: 5.2776 - learning_rate: 0.0010
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 45.3458 - mae: 4.6013 - val_loss: 51.6728 - val_mae: 4.9781 - learning_rate: 0.0010
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 44.0664 - mae: 4.5404 - val_loss: 46.7139 - val_mae: 4.6984 - learning_rate: 0.0010
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 42.4154 - mae: 4.4559 - val_loss: 46.8304 - val_mae: 4.7534 - learning_rate: 0.0010
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 40.4253 - mae: 4.3350 - val_loss: 45.8820 - val_mae: 4.6745 - learning_rate: 0.0010
Epoch 12/100 ━━━━━━━━━━━━━━━━━━━━ loss: 39.4400 - mae: 4.3073 - val_loss: 44.3116 - val_mae: 4.6096 - learning_rate: 0.0010
Epoch 13/100 ━━━━━━━━━━━━━━━━━━━━ loss: 38.1006 - mae: 4.2348 - val_loss: 42.7258 - val_mae: 4.5349 - learning_rate: 0.0010
Epoch 14/100 ━━━━━━━━━━━━━━━━━━━━ loss: 36.7023 - mae: 4.1434 - val_loss: 43.0890 - val_mae: 4.4592 - learning_rate: 0.0010
Epoch 15/100 ━━━━━━━━━━━━━━━━━━━━ loss: 35.8021 - mae: 4.1046 - val_loss: 43.2633 - val_mae: 4.4907 - learning_rate: 0.0010
Epoch 16/100 ━━━━━━━━━━━━━━━━━━━━ loss: 34.5885 - mae: 4.0211 - val_loss: 43.1467 - val_mae: 4.5261 - learning_rate: 0.0010
Epoch 17/100 ━━━━━━━━━━━━━━━━━━━━ loss: 33.4114 - mae: 3.9651 - val_loss: 42.7900 - val_mae: 4.4395 - learning_rate: 0.0010
Epoch 18/100 ━━━━━━━━━━━━━━━━━━━━ loss: 32.5362 - mae: 3.9260 - val_loss: 40.5797 - val_mae: 4.3576 - learning_rate: 0.0010
Epoch 19/100 ━━━━━━━━━━━━━━━━━━━━ loss: 31.2205 - mae: 3.8459 - val_loss: 43.1901 - val_mae: 4.4603 - learning_rate: 0.0010
Epoch 20/100 ━━━━━━━━━━━━━━━━━━━━ loss: 30.1846 - mae: 3.7884 - val_loss: 40.9200 - val_mae: 4.3352 - learning_rate: 0.0010
Epoch 21/100 ━━━━━━━━━━━━━━━━━━━━ loss: 29.4932 - mae: 3.7460 - val_loss: 41.9323 - val_mae: 4.3729 - learning_rate: 0.0010
Epoch 22/100 ━━━━━━━━━━━━━━━━━━━━ loss: 28.1724 - mae: 3.6684 - val_loss: 41.1825 - val_mae: 4.3898 - learning_rate: 0.0010
Epoch 23/100 ━━━━━━━━━━━━━━━━━━━━ loss: 27.2323 - mae: 3.6063 - val_loss: 39.6626 - val_mae: 4.2689 - learning_rate: 0.0010
Epoch 24/100 ━━━━━━━━━━━━━━━━━━━━ loss: 26.5389 - mae: 3.5704 - val_loss: 40.2890 - val_mae: 4.3088 - learning_rate: 0.0010
Epoch 25/100 ━━━━━━━━━━━━━━━━━━━━ loss: 25.5003 - mae: 3.5046 - val_loss: 41.0627 - val_mae: 4.3369 - learning_rate: 0.0010
Epoch 26/100 ━━━━━━━━━━━━━━━━━━━━ loss: 24.6796 - mae: 3.4500 - val_loss: 40.6633 - val_mae: 4.3549 - learning_rate: 0.0010
Epoch 27/100 ━━━━━━━━━━━━━━━━━━━━ loss: 23.9718 - mae: 3.4023 - val_loss: 40.9918 - val_mae: 4.3957 - learning_rate: 0.0010
Epoch 28/100 ━━━━━━━━━━━━━━━━━━━━ loss: 23.2976 - mae: 3.3558 - val_loss: 41.7845 - val_mae: 4.3899 - learning_rate: 0.0010
Learning rate reduced: 0.001000 -> 0.000500
Epoch 29/100 ━━━━━━━━━━━━━━━━━━━━ loss: 20.0043 - mae: 3.1044 - val_loss: 39.9739 - val_mae: 4.3023 - learning_rate: 0.0005
Epoch 30/100 ━━━━━━━━━━━━━━━━━━━━ loss: 19.3409 - mae: 3.0603 - val_loss: 40.4239 - val_mae: 4.3110 - learning_rate: 0.0005
Epoch 31/100 ━━━━━━━━━━━━━━━━━━━━ loss: 18.8290 - mae: 3.0192 - val_loss: 40.4111 - val_mae: 4.3259 - learning_rate: 0.0005
Epoch 32/100 ━━━━━━━━━━━━━━━━━━━━ loss: 18.4254 - mae: 2.9830 - val_loss: 41.7662 - val_mae: 4.3602 - learning_rate: 0.0005
Epoch 33/100 ━━━━━━━━━━━━━━━━━━━━ loss: 17.9740 - mae: 2.9499 - val_loss: 41.1366 - val_mae: 4.3294 - learning_rate: 0.0005
Learning rate reduced: 0.000500 -> 0.000250
```

## 원본 모델 결과(Output_SpCas9_mae: 0.0432)와 거의 유사한 성능 재현
