# 데이터셋 구성에 따른 성능 변화 테스트
* A375 viability, MOLM13 viability 두 가지 데이터의 점수를 평균 낸 데이터 셋(약 1800개)을 사용하여 파인 튜닝 진행

```text
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
```

## 결과(val_mae: 0.1812)가 기존 성능(val_mae: 0.1342)보다 떨어짐

