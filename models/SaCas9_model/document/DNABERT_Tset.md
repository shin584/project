# DNABERT 임베딩 벡터를 활용 테스트
## SpCas9 데이터셋의 일반 서열과 DNABERT 임베딩 벡터 각각을 활용한 학습 결과 비교를 위한 테스트 시행
---
* 1. 일반 서열 학습 결과
```text
Selected adapter: onehot
Total parameters: 141249
Weights will be saved to: /content/drive/MyDrive/MC_project/crispr_cleavage_predictor/data
One-hot tensor shape: torch.Size([36445, 30, 4])
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 482.3380 - mae: 19.2335 - val_loss: 413.8894 - val_mae: 17.8036 - learning_rate: 0.0010
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 412.5929 - mae: 17.6762 - val_loss: 413.5651 - val_mae: 17.7426 - learning_rate: 0.0010
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 217.3845 - mae: 11.0336 - val_loss: 85.8481 - val_mae: 6.2973 - learning_rate: 0.0010
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 66.3291 - mae: 5.6066 - val_loss: 55.6043 - val_mae: 5.1405 - learning_rate: 0.0010
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 53.9150 - mae: 5.0471 - val_loss: 51.3568 - val_mae: 4.9366 - learning_rate: 0.0010
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 49.3770 - mae: 4.8181 - val_loss: 51.8261 - val_mae: 4.9409 - learning_rate: 0.0010
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 46.7216 - mae: 4.6759 - val_loss: 46.7987 - val_mae: 4.6763 - learning_rate: 0.0010
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 44.1060 - mae: 4.5467 - val_loss: 51.3869 - val_mae: 4.7752 - learning_rate: 0.0010
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 42.1714 - mae: 4.4394 - val_loss: 45.4307 - val_mae: 4.5934 - learning_rate: 0.0010
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 40.2747 - mae: 4.3238 - val_loss: 45.0968 - val_mae: 4.5902 - learning_rate: 0.0010
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 38.9363 - mae: 4.2622 - val_loss: 43.3883 - val_mae: 4.4824 - learning_rate: 0.0010
Epoch 12/100 ━━━━━━━━━━━━━━━━━━━━ loss: 37.0720 - mae: 4.1633 - val_loss: 44.0502 - val_mae: 4.4542 - learning_rate: 0.0010
Epoch 13/100 ━━━━━━━━━━━━━━━━━━━━ loss: 35.8490 - mae: 4.1032 - val_loss: 42.4530 - val_mae: 4.4069 - learning_rate: 0.0010
Epoch 14/100 ━━━━━━━━━━━━━━━━━━━━ loss: 34.5179 - mae: 4.0355 - val_loss: 43.9673 - val_mae: 4.4983 - learning_rate: 0.0010
Epoch 15/100 ━━━━━━━━━━━━━━━━━━━━ loss: 33.2530 - mae: 3.9586 - val_loss: 41.6633 - val_mae: 4.3927 - learning_rate: 0.0010
Epoch 16/100 ━━━━━━━━━━━━━━━━━━━━ loss: 32.0780 - mae: 3.8923 - val_loss: 41.4141 - val_mae: 4.4285 - learning_rate: 0.0010
Epoch 17/100 ━━━━━━━━━━━━━━━━━━━━ loss: 30.8769 - mae: 3.8234 - val_loss: 40.7778 - val_mae: 4.3151 - learning_rate: 0.0010
Epoch 18/100 ━━━━━━━━━━━━━━━━━━━━ loss: 29.8315 - mae: 3.7724 - val_loss: 41.4026 - val_mae: 4.3730 - learning_rate: 0.0010
Epoch 19/100 ━━━━━━━━━━━━━━━━━━━━ loss: 28.5642 - mae: 3.6935 - val_loss: 40.9038 - val_mae: 4.3377 - learning_rate: 0.0010
Epoch 20/100 ━━━━━━━━━━━━━━━━━━━━ loss: 27.2617 - mae: 3.6060 - val_loss: 41.1034 - val_mae: 4.3444 - learning_rate: 0.0010
Epoch 21/100 ━━━━━━━━━━━━━━━━━━━━ loss: 26.5445 - mae: 3.5729 - val_loss: 42.3112 - val_mae: 4.4567 - learning_rate: 0.0010
Epoch 22/100 ━━━━━━━━━━━━━━━━━━━━ loss: 25.4563 - mae: 3.5066 - val_loss: 40.6055 - val_mae: 4.3200 - learning_rate: 0.0010
Epoch 23/100 ━━━━━━━━━━━━━━━━━━━━ loss: 24.6289 - mae: 3.4356 - val_loss: 42.3934 - val_mae: 4.4578 - learning_rate: 0.0010
Epoch 24/100 ━━━━━━━━━━━━━━━━━━━━ loss: 23.5477 - mae: 3.3587 - val_loss: 41.2958 - val_mae: 4.3541 - learning_rate: 0.0010
Epoch 25/100 ━━━━━━━━━━━━━━━━━━━━ loss: 22.7333 - mae: 3.3125 - val_loss: 41.6931 - val_mae: 4.3661 - learning_rate: 0.0010
Epoch 26/100 ━━━━━━━━━━━━━━━━━━━━ loss: 21.9278 - mae: 3.2498 - val_loss: 41.1044 - val_mae: 4.3677 - learning_rate: 0.0010
Epoch 27/100 ━━━━━━━━━━━━━━━━━━━━ loss: 21.2024 - mae: 3.1924 - val_loss: 40.9707 - val_mae: 4.3507 - learning_rate: 0.0010
Learning rate reduced: 0.001000 -> 0.000500
Epoch 28/100 ━━━━━━━━━━━━━━━━━━━━ loss: 18.2507 - mae: 2.9605 - val_loss: 40.7349 - val_mae: 4.3238 - learning_rate: 0.0005
Epoch 29/100 ━━━━━━━━━━━━━━━━━━━━ loss: 17.4280 - mae: 2.8951 - val_loss: 40.3325 - val_mae: 4.3031 - learning_rate: 0.0005
Epoch 30/100 ━━━━━━━━━━━━━━━━━━━━ loss: 17.0021 - mae: 2.8604 - val_loss: 40.7601 - val_mae: 4.3398 - learning_rate: 0.0005
Epoch 31/100 ━━━━━━━━━━━━━━━━━━━━ loss: 16.6913 - mae: 2.8281 - val_loss: 41.1805 - val_mae: 4.3229 - learning_rate: 0.0005
Epoch 32/100 ━━━━━━━━━━━━━━━━━━━━ loss: 16.1779 - mae: 2.7830 - val_loss: 41.6224 - val_mae: 4.3588 - learning_rate: 0.0005
Epoch 33/100 ━━━━━━━━━━━━━━━━━━━━ loss: 15.8713 - mae: 2.7558 - val_loss: 42.0833 - val_mae: 4.3670 - learning_rate: 0.0005
Epoch 34/100 ━━━━━━━━━━━━━━━━━━━━ loss: 15.5310 - mae: 2.7274 - val_loss: 41.2736 - val_mae: 4.3775 - learning_rate: 0.0005
Learning rate reduced: 0.000500 -> 0.000250
Epoch 35/100 ━━━━━━━━━━━━━━━━━━━━ loss: 14.0307 - mae: 2.5828 - val_loss: 41.2619 - val_mae: 4.3590 - learning_rate: 0.0003
Epoch 36/100 ━━━━━━━━━━━━━━━━━━━━ loss: 13.6806 - mae: 2.5424 - val_loss: 42.3466 - val_mae: 4.3888 - learning_rate: 0.0003
Epoch 37/100 ━━━━━━━━━━━━━━━━━━━━ loss: 13.4896 - mae: 2.5245 - val_loss: 42.2701 - val_mae: 4.4025 - learning_rate: 0.0003
Epoch 38/100 ━━━━━━━━━━━━━━━━━━━━ loss: 13.3374 - mae: 2.5080 - val_loss: 42.3814 - val_mae: 4.4029 - learning_rate: 0.0003
Epoch 39/100 ━━━━━━━━━━━━━━━━━━━━ loss: 13.1642 - mae: 2.4867 - val_loss: 42.2549 - val_mae: 4.4002 - learning_rate: 0.0003
Learning rate reduced: 0.000250 -> 0.000125
```
* 최종 결과 : loss: 17.4280 - mae: 2.8951 - val_loss: 40.3325 - val_mae: 4.3031 - learning_rate: 0.0005

---
*  DNABERT 임베딩 벡터 학습 결과
*  SpCas9 서열을 DNABERT 모델에 통과시켜 얻은 임베딩 벡터로 학습 시도
```text
Selected adapter: dnabert
Total parameters: 239041
Weights will be saved to: /content/drive/MyDrive/MC_project/crispr_cleavage_predictor/data
Loading pre-computed tensor data from: /content/drive/MyDrive/MC_project/crispr_cleavage_predictor/dataset/spcas9_pretrained_features.pt
One-hot tensor shape: torch.Size([36445, 768])
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0419 - mae: 0.1767 - val_loss: 0.0404 - val_mae: 0.1676 - learning_rate: 0.0010
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0370 - mae: 0.1615 - val_loss: 0.0382 - val_mae: 0.1628 - learning_rate: 0.0010
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0356 - mae: 0.1568 - val_loss: 0.0370 - val_mae: 0.1621 - learning_rate: 0.0010
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0344 - mae: 0.1527 - val_loss: 0.0359 - val_mae: 0.1605 - learning_rate: 0.0010
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0335 - mae: 0.1498 - val_loss: 0.0366 - val_mae: 0.1583 - learning_rate: 0.0010
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0327 - mae: 0.1472 - val_loss: 0.0358 - val_mae: 0.1516 - learning_rate: 0.0010
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0317 - mae: 0.1433 - val_loss: 0.0357 - val_mae: 0.1534 - learning_rate: 0.0010
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0307 - mae: 0.1404 - val_loss: 0.0361 - val_mae: 0.1514 - learning_rate: 0.0010
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0300 - mae: 0.1383 - val_loss: 0.0351 - val_mae: 0.1515 - learning_rate: 0.0010
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0290 - mae: 0.1348 - val_loss: 0.0354 - val_mae: 0.1509 - learning_rate: 0.0010
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0284 - mae: 0.1325 - val_loss: 0.0351 - val_mae: 0.1495 - learning_rate: 0.0010
Epoch 12/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0273 - mae: 0.1291 - val_loss: 0.0353 - val_mae: 0.1446 - learning_rate: 0.0010
Epoch 13/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0263 - mae: 0.1258 - val_loss: 0.0358 - val_mae: 0.1451 - learning_rate: 0.0010
Epoch 14/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0258 - mae: 0.1243 - val_loss: 0.0351 - val_mae: 0.1435 - learning_rate: 0.0010
Epoch 15/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0251 - mae: 0.1218 - val_loss: 0.0342 - val_mae: 0.1440 - learning_rate: 0.0010
Epoch 16/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0240 - mae: 0.1185 - val_loss: 0.0357 - val_mae: 0.1419 - learning_rate: 0.0010
Epoch 17/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0231 - mae: 0.1154 - val_loss: 0.0358 - val_mae: 0.1417 - learning_rate: 0.0010
Epoch 18/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0223 - mae: 0.1126 - val_loss: 0.0364 - val_mae: 0.1452 - learning_rate: 0.0010
Epoch 19/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0219 - mae: 0.1112 - val_loss: 0.0350 - val_mae: 0.1417 - learning_rate: 0.0010
Epoch 20/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0208 - mae: 0.1075 - val_loss: 0.0344 - val_mae: 0.1388 - learning_rate: 0.0010
Learning rate reduced: 0.001000 -> 0.000500
Epoch 21/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0181 - mae: 0.0990 - val_loss: 0.0351 - val_mae: 0.1345 - learning_rate: 0.0005
Epoch 22/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0171 - mae: 0.0955 - val_loss: 0.0346 - val_mae: 0.1364 - learning_rate: 0.0005
Epoch 23/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0166 - mae: 0.0941 - val_loss: 0.0349 - val_mae: 0.1363 - learning_rate: 0.0005
Epoch 24/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0161 - mae: 0.0920 - val_loss: 0.0346 - val_mae: 0.1349 - learning_rate: 0.0005
Epoch 25/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0153 - mae: 0.0898 - val_loss: 0.0364 - val_mae: 0.1398 - learning_rate: 0.0005
Learning rate reduced: 0.000500 -> 0.000250
```
* 최종 결과 : loss: 0.0181 - mae: 0.0990 - val_loss: 0.0351 - val_mae: 0.1345 - learning_rate: 0.0005
---
## 결론
* 오히려 성능이 하락함
* 원인 분석
  1. 36bp 입력을 768개의 피처로 변환하면서 CRISPR-CAS에서 중요한 위치정보가 소실됨
  2. 36bp 입력이 768개의 피처로 변환되면서 파라미터 수가 급증하여 과적합 발생
## 학습에 부적합하다고 판단되어 DNABERT 계획 보류


