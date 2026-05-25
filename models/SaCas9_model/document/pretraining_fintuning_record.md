1. pretraining

- SpCas9 데이터셋을 이용한 프리트레이닝 결과

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

    - 원본 모델 결과(Output_SpCas9_mae: 0.0432)와 거의 유사한 성능 재현

2. finetuning

- A375 viability, MOLM13 viability 두가지 데이터의 점수를 평균낸 데이터셋(약1800개)을 사용하여 파인튜닝 진행

(crispr_env) C:\project\MC_project\SaCas9\crispr-cleavage-predictor>python main.py --mode finetune --data_path "dataset/SaCas9_Averaged_Training_Data.csv" --epochs 100 --batch_size 64 --weight_path "data/spcas9_pretrained_weight.pth"
Loading sequence data from: dataset/SaCas9_Averaged_Training_Data.csv
Dataset successfully loaded. Feature shape: torch.Size([1805, 36, 4])
Building One-hot CNN+RNN Backbone...
Total trainable parameters: 141249
Loading pretrained weights from: data/spcas9_pretrained_weight.pth
Starting training on local environment...
Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.1301 - mae: 0.2755 - val_loss: 0.1073 - val_mae: 0.2578 - learning_rate: 0.0001
Epoch 2/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0885 - mae: 0.2365 - val_loss: 0.0957 - val_mae: 0.2471 - learning_rate: 0.0001
Epoch 3/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0775 - mae: 0.2214 - val_loss: 0.0881 - val_mae: 0.2382 - learning_rate: 0.0001
Epoch 4/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0694 - mae: 0.2104 - val_loss: 0.0827 - val_mae: 0.2306 - learning_rate: 0.0001
Epoch 5/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0626 - mae: 0.2000 - val_loss: 0.0808 - val_mae: 0.2305 - learning_rate: 0.0001
Epoch 6/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0591 - mae: 0.1950 - val_loss: 0.0756 - val_mae: 0.2218 - learning_rate: 0.0001
Epoch 7/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0554 - mae: 0.1879 - val_loss: 0.0737 - val_mae: 0.2186 - learning_rate: 0.0001
Epoch 8/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0529 - mae: 0.1835 - val_loss: 0.0707 - val_mae: 0.2149 - learning_rate: 0.0001
Epoch 9/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0490 - mae: 0.1772 - val_loss: 0.0681 - val_mae: 0.2108 - learning_rate: 0.0001
Epoch 10/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0473 - mae: 0.1740 - val_loss: 0.0673 - val_mae: 0.2082 - learning_rate: 0.0001
Epoch 11/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0451 - mae: 0.1699 - val_loss: 0.0672 - val_mae: 0.2103 - learning_rate: 0.0001
Epoch 12/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0437 - mae: 0.1672 - val_loss: 0.0639 - val_mae: 0.2040 - learning_rate: 0.0001
Epoch 13/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0422 - mae: 0.1642 - val_loss: 0.0628 - val_mae: 0.2025 - learning_rate: 0.0001
Epoch 14/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0414 - mae: 0.1631 - val_loss: 0.0632 - val_mae: 0.2034 - learning_rate: 0.0001
Epoch 15/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0402 - mae: 0.1600 - val_loss: 0.0629 - val_mae: 0.2023 - learning_rate: 0.0001
Epoch 16/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0400 - mae: 0.1598 - val_loss: 0.0606 - val_mae: 0.1992 - learning_rate: 0.0001
Epoch 17/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0372 - mae: 0.1541 - val_loss: 0.0590 - val_mae: 0.1958 - learning_rate: 0.0001
Epoch 18/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0370 - mae: 0.1537 - val_loss: 0.0582 - val_mae: 0.1953 - learning_rate: 0.0001
Epoch 19/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0354 - mae: 0.1502 - val_loss: 0.0580 - val_mae: 0.1941 - learning_rate: 0.0001
Epoch 20/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0355 - mae: 0.1506 - val_loss: 0.0577 - val_mae: 0.1946 - learning_rate: 0.0001
Epoch 21/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0346 - mae: 0.1494 - val_loss: 0.0565 - val_mae: 0.1924 - learning_rate: 0.0001
Epoch 22/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0336 - mae: 0.1468 - val_loss: 0.0562 - val_mae: 0.1915 - learning_rate: 0.0001
Epoch 23/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0324 - mae: 0.1442 - val_loss: 0.0555 - val_mae: 0.1906 - learning_rate: 0.0001
Epoch 24/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0323 - mae: 0.1443 - val_loss: 0.0561 - val_mae: 0.1907 - learning_rate: 0.0001
Epoch 25/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0318 - mae: 0.1427 - val_loss: 0.0551 - val_mae: 0.1898 - learning_rate: 0.0001
Epoch 26/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0313 - mae: 0.1416 - val_loss: 0.0561 - val_mae: 0.1909 - learning_rate: 0.0001
Epoch 27/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0301 - mae: 0.1394 - val_loss: 0.0558 - val_mae: 0.1914 - learning_rate: 0.0001
Epoch 28/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0299 - mae: 0.1384 - val_loss: 0.0547 - val_mae: 0.1880 - learning_rate: 0.0001
Epoch 29/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0298 - mae: 0.1378 - val_loss: 0.0544 - val_mae: 0.1876 - learning_rate: 0.0001
Epoch 30/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0290 - mae: 0.1368 - val_loss: 0.0539 - val_mae: 0.1870 - learning_rate: 0.0001
Epoch 31/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0282 - mae: 0.1350 - val_loss: 0.0541 - val_mae: 0.1875 - learning_rate: 0.0001
Epoch 32/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0278 - mae: 0.1340 - val_loss: 0.0533 - val_mae: 0.1855 - learning_rate: 0.0001
Epoch 33/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0274 - mae: 0.1326 - val_loss: 0.0531 - val_mae: 0.1856 - learning_rate: 0.0001
Epoch 34/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0268 - mae: 0.1308 - val_loss: 0.0531 - val_mae: 0.1857 - learning_rate: 0.0001
Epoch 35/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0265 - mae: 0.1312 - val_loss: 0.0526 - val_mae: 0.1851 - learning_rate: 0.0001
Epoch 36/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0261 - mae: 0.1292 - val_loss: 0.0549 - val_mae: 0.1898 - learning_rate: 0.0001
Epoch 37/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0260 - mae: 0.1288 - val_loss: 0.0528 - val_mae: 0.1850 - learning_rate: 0.0001
Epoch 38/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0250 - mae: 0.1264 - val_loss: 0.0533 - val_mae: 0.1867 - learning_rate: 0.0001
Epoch 39/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0249 - mae: 0.1263 - val_loss: 0.0528 - val_mae: 0.1856 - learning_rate: 0.0001
Epoch 40/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0250 - mae: 0.1269 - val_loss: 0.0521 - val_mae: 0.1837 - learning_rate: 0.0001
Epoch 41/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0239 - mae: 0.1236 - val_loss: 0.0534 - val_mae: 0.1858 - learning_rate: 0.0001
Epoch 42/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0239 - mae: 0.1240 - val_loss: 0.0519 - val_mae: 0.1835 - learning_rate: 0.0001
Epoch 43/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0232 - mae: 0.1218 - val_loss: 0.0518 - val_mae: 0.1844 - learning_rate: 0.0001
Epoch 44/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0233 - mae: 0.1216 - val_loss: 0.0520 - val_mae: 0.1848 - learning_rate: 0.0001
Epoch 45/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0227 - mae: 0.1207 - val_loss: 0.0517 - val_mae: 0.1836 - learning_rate: 0.0001
Epoch 46/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0221 - mae: 0.1183 - val_loss: 0.0515 - val_mae: 0.1834 - learning_rate: 0.0001
Epoch 47/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0220 - mae: 0.1184 - val_loss: 0.0513 - val_mae: 0.1833 - learning_rate: 0.0001
Epoch 48/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0213 - mae: 0.1163 - val_loss: 0.0514 - val_mae: 0.1823 - learning_rate: 0.0001
Epoch 49/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0213 - mae: 0.1162 - val_loss: 0.0512 - val_mae: 0.1827 - learning_rate: 0.0001
Epoch 50/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0209 - mae: 0.1152 - val_loss: 0.0507 - val_mae: 0.1825 - learning_rate: 0.0001
Epoch 51/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0207 - mae: 0.1144 - val_loss: 0.0515 - val_mae: 0.1834 - learning_rate: 0.0001
Epoch 52/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0204 - mae: 0.1133 - val_loss: 0.0514 - val_mae: 0.1827 - learning_rate: 0.0001
Epoch 53/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0204 - mae: 0.1139 - val_loss: 0.0509 - val_mae: 0.1830 - learning_rate: 0.0001
Epoch 54/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0198 - mae: 0.1125 - val_loss: 0.0512 - val_mae: 0.1826 - learning_rate: 0.0001
Epoch 55/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0198 - mae: 0.1116 - val_loss: 0.0510 - val_mae: 0.1830 - learning_rate: 0.0001
Learning rate reduced: 0.000100 -> 0.000050
Epoch 56/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0191 - mae: 0.1096 - val_loss: 0.0505 - val_mae: 0.1818 - learning_rate: 0.0001
Epoch 57/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0190 - mae: 0.1097 - val_loss: 0.0506 - val_mae: 0.1820 - learning_rate: 0.0001
Epoch 58/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0188 - mae: 0.1092 - val_loss: 0.0504 - val_mae: 0.1819 - learning_rate: 0.0001
Epoch 59/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0186 - mae: 0.1085 - val_loss: 0.0507 - val_mae: 0.1824 - learning_rate: 0.0001
Epoch 60/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0186 - mae: 0.1080 - val_loss: 0.0507 - val_mae: 0.1821 - learning_rate: 0.0001
Epoch 61/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0185 - mae: 0.1078 - val_loss: 0.0510 - val_mae: 0.1826 - learning_rate: 0.0001
Epoch 62/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0183 - mae: 0.1075 - val_loss: 0.0507 - val_mae: 0.1819 - learning_rate: 0.0001
Epoch 63/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0179 - mae: 0.1061 - val_loss: 0.0502 - val_mae: 0.1814 - learning_rate: 0.0001
Epoch 64/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0176 - mae: 0.1054 - val_loss: 0.0503 - val_mae: 0.1817 - learning_rate: 0.0001
Epoch 65/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0178 - mae: 0.1058 - val_loss: 0.0501 - val_mae: 0.1813 - learning_rate: 0.0001
Epoch 66/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0176 - mae: 0.1055 - val_loss: 0.0502 - val_mae: 0.1813 - learning_rate: 0.0001
Epoch 67/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0176 - mae: 0.1051 - val_loss: 0.0504 - val_mae: 0.1819 - learning_rate: 0.0001
Epoch 68/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0176 - mae: 0.1052 - val_loss: 0.0501 - val_mae: 0.1811 - learning_rate: 0.0001
Epoch 69/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0172 - mae: 0.1038 - val_loss: 0.0500 - val_mae: 0.1809 - learning_rate: 0.0001
Epoch 70/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0171 - mae: 0.1036 - val_loss: 0.0500 - val_mae: 0.1810 - learning_rate: 0.0001
Epoch 71/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0170 - mae: 0.1029 - val_loss: 0.0500 - val_mae: 0.1811 - learning_rate: 0.0001
Epoch 72/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0171 - mae: 0.1035 - val_loss: 0.0503 - val_mae: 0.1813 - learning_rate: 0.0001
Epoch 73/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0167 - mae: 0.1021 - val_loss: 0.0500 - val_mae: 0.1815 - learning_rate: 0.0001
Epoch 74/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0165 - mae: 0.1017 - val_loss: 0.0500 - val_mae: 0.1814 - learning_rate: 0.0001
Epoch 75/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0166 - mae: 0.1017 - val_loss: 0.0502 - val_mae: 0.1813 - learning_rate: 0.0001
Epoch 76/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0164 - mae: 0.1011 - val_loss: 0.0499 - val_mae: 0.1810 - learning_rate: 0.0001
Epoch 77/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0162 - mae: 0.1006 - val_loss: 0.0502 - val_mae: 0.1819 - learning_rate: 0.0001
Epoch 78/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0161 - mae: 0.1007 - val_loss: 0.0499 - val_mae: 0.1808 - learning_rate: 0.0001
Epoch 79/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0160 - mae: 0.0996 - val_loss: 0.0500 - val_mae: 0.1812 - learning_rate: 0.0001
Epoch 80/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0158 - mae: 0.0992 - val_loss: 0.0502 - val_mae: 0.1817 - learning_rate: 0.0001
Epoch 81/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0157 - mae: 0.0987 - val_loss: 0.0501 - val_mae: 0.1811 - learning_rate: 0.0001
Learning rate reduced: 0.000050 -> 0.000025
Epoch 82/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0155 - mae: 0.0980 - val_loss: 0.0504 - val_mae: 0.1823 - learning_rate: 0.0000
Epoch 83/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0156 - mae: 0.0983 - val_loss: 0.0500 - val_mae: 0.1811 - learning_rate: 0.0000
Epoch 84/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0154 - mae: 0.0978 - val_loss: 0.0501 - val_mae: 0.1814 - learning_rate: 0.0000
Epoch 85/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0150 - mae: 0.0970 - val_loss: 0.0500 - val_mae: 0.1811 - learning_rate: 0.0000
Epoch 86/100 ━━━━━━━━━━━━━━━━━━━━ loss: 0.0152 - mae: 0.0970 - val_loss: 0.0500 - val_mae: 0.1812 - learning_rate: 0.0000
Learning rate reduced: 0.000025 -> 0.000013

    - 결과(val_mae: 0.2976)가 기준 성능(val_mae: 0.1342)에 미치지 못함

- SaCas9의 원본 데이터셋(약 7000여개)를 모두 사용



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

    - 성능이 오히려 하락함

- 기준 모델에서 사용된 데이터(A375 viability, MOLM13 viabilit)를 사용하여 다시 시도(약 3600여개)
    - 기준 모델 데이터셋 + 파인튜닝

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

    - 기준 성능 초과 달성 (val_mae: 0.1342 -> val_mae: 0.1260)

- 기준 모델에서 사용된 데이터(A375 viability, MOLM13 viabilit)에 293T viability를 추가하여 시도(약 5000여개)

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

- 조기 과적합 현상을 잡기 위해 모델 고도화 작업 수행
    - dropout rate를 0.4로 높임
    - Loss 함수를 `MSELoss`에서 `HuberLoss`로 교체
    - K-Fold Cross Validation 도입
    - test set 분할 및 평가 기능 도입


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

3. 절단 효율 예측
- 훈련된 모델에 서열을 입력하여 절단 효율 출력
(crispr_env) C:\project\MC_project\SaCas9\crispr-cleavage-predictor>python predict.py --weight_path "data/best_model_fold1.pth" --sequence "GGATCTGGTCTACCGTGAAGTTCACCTGGGCAAGAC"

Target Sequence : GGATCTGGTCTACCGTGAAGTTCACCTGGGCAAGAC
Sequence Length : 36 bp
Loading model weights from: data/best_model_fold1.pth

Predicted Cleavage Efficiency Score: 0.7131