# chai1 trunk recycle 옵션 조절 시도

## recycles = 5

* **1차 시기**

```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 5/5 [01:29<00:00, 17.91s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6321, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 13:29 .
drwxrwxr-x 3 shin shin   39 May  1 13:26 ..
-rw-rw-r-- 1 shin shin 832K May  1 13:29 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 13:29 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.632065, max=0.632065, mean=0.632065
ptm: shape=(1,), min=0.661679, max=0.661679, mean=0.661679
iptm: shape=(1,), min=0.624661, max=0.624661, mean=0.624661
per_chain_ptm: shape=(1, 4), min=0.266760, max=0.744890, mean=0.503324
per_chain_pair_iptm: shape=(1, 4, 4), min=0.100190, max=0.744890, mean=0.407327
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=4.000000, mean=0.875000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6247
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1033 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1033, 'iptm': 0.624660849571228}
```
* **결과 :** ipTM은 증가하였으나 contact score의 편차가 큼.


## recycle = 6
* **1차 시기**

```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 6/6 [01:47<00:00, 17.94s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6468, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 13:36 .
drwxrwxr-x 3 shin shin   39 May  1 13:32 ..
-rw-rw-r-- 1 shin shin 831K May  1 13:36 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 13:36 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.646811, max=0.646811, mean=0.646811
ptm: shape=(1,), min=0.673582, max=0.673582, mean=0.673582
iptm: shape=(1,), min=0.640118, max=0.640118, mean=0.640118
per_chain_ptm: shape=(1, 4), min=0.275540, max=0.745039, mean=0.510311
per_chain_pair_iptm: shape=(1, 4, 4), min=0.087594, max=0.745039, mean=0.412888
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=3.000000, mean=0.750000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6401
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1286 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1286, 'iptm': 0.6401177644729614}
```

* **2차 시기**
```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 6/6 [01:47<00:00, 17.95s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6421, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 13:57 .
drwxrwxr-x 3 shin shin   39 May  1 13:54 ..
-rw-rw-r-- 1 shin shin 832K May  1 13:57 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 13:57 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.642139, max=0.642139, mean=0.642139
ptm: shape=(1,), min=0.670469, max=0.670469, mean=0.670469
iptm: shape=(1,), min=0.635057, max=0.635057, mean=0.635057
per_chain_ptm: shape=(1, 4), min=0.274298, max=0.745519, mean=0.510180
per_chain_pair_iptm: shape=(1, 4, 4), min=0.097029, max=0.745519, mean=0.414911
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=17.000000, mean=2.937500
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6351
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1555 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1555, 'iptm': 0.6350568532943726}
```

* **3차시기**

```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 6/6 [01:47<00:00, 17.95s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6441, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 14:03 .
drwxrwxr-x 3 shin shin   39 May  1 13:59 ..
-rw-rw-r-- 1 shin shin 832K May  1 14:03 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 14:03 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.644106, max=0.644106, mean=0.644106
ptm: shape=(1,), min=0.672197, max=0.672197, mean=0.672197
iptm: shape=(1,), min=0.637083, max=0.637083, mean=0.637083
per_chain_ptm: shape=(1, 4), min=0.275068, max=0.745689, mean=0.510920
per_chain_pair_iptm: shape=(1, 4, 4), min=0.089362, max=0.745689, mean=0.413870
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=5.000000, mean=1.562500
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6371
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1115 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1115, 'iptm': 0.6370832920074463}
```

## recycles = 7

```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 7/7 [02:05<00:00, 17.97s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6618, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 13:41 .
drwxrwxr-x 3 shin shin   39 May  1 13:37 ..
-rw-rw-r-- 1 shin shin 832K May  1 13:41 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 13:41 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.661825, max=0.661825, mean=0.661825
ptm: shape=(1,), min=0.688991, max=0.688991, mean=0.688991
iptm: shape=(1,), min=0.655034, max=0.655034, mean=0.655034
per_chain_ptm: shape=(1, 4), min=0.269775, max=0.745996, mean=0.515859
per_chain_pair_iptm: shape=(1, 4, 4), min=0.100074, max=0.745996, mean=0.425768
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=5.000000, mean=1.250000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6550
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1006 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1006, 'iptm': 0.6550337672233582}
```

* **2차시기**
```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 7/7 [02:05<00:00, 17.98s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6582, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 14:08 .
drwxrwxr-x 3 shin shin   39 May  1 14:03 ..
-rw-rw-r-- 1 shin shin 832K May  1 14:08 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 14:08 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.658160, max=0.658160, mean=0.658160
ptm: shape=(1,), min=0.683878, max=0.683878, mean=0.683878
iptm: shape=(1,), min=0.651730, max=0.651730, mean=0.651730
per_chain_ptm: shape=(1, 4), min=0.277983, max=0.746406, mean=0.517240
per_chain_pair_iptm: shape=(1, 4, 4), min=0.094284, max=0.746406, mean=0.423630
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=6.000000, mean=1.250000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6517
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1213 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1213, 'iptm': 0.6517298221588135}
```

* **3차시기**
```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 7/7 [02:05<00:00, 17.97s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6521, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 14:13 .
drwxrwxr-x 3 shin shin   39 May  1 14:09 ..
-rw-rw-r-- 1 shin shin 831K May  1 14:13 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 14:13 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.652083, max=0.652083, mean=0.652083
ptm: shape=(1,), min=0.678702, max=0.678702, mean=0.678702
iptm: shape=(1,), min=0.645428, max=0.645428, mean=0.645428
per_chain_ptm: shape=(1, 4), min=0.278219, max=0.746407, mean=0.516350
per_chain_pair_iptm: shape=(1, 4, 4), min=0.096944, max=0.746407, mean=0.423010
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=9.000000, mean=1.750000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6454
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1116 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1116, 'iptm': 0.6454284191131592}
```

* **4차 시기**

```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 7/7 [02:05<00:00, 17.98s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6546, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 14:18 .
drwxrwxr-x 3 shin shin   39 May  1 14:14 ..
-rw-rw-r-- 1 shin shin 832K May  1 14:18 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 14:18 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.654602, max=0.654602, mean=0.654602
ptm: shape=(1,), min=0.680670, max=0.680670, mean=0.680670
iptm: shape=(1,), min=0.648085, max=0.648085, mean=0.648085
per_chain_ptm: shape=(1, 4), min=0.276452, max=0.746266, mean=0.516627
per_chain_pair_iptm: shape=(1, 4, 4), min=0.103059, max=0.746266, mean=0.424076
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=8.000000, mean=2.000000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6481
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 999 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 999, 'iptm': 0.6480850577354431}
(chai1) shin@TeamB511105:~$
```

* **5차시기**
```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 7/7 [02:05<00:00, 17.97s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6469, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 14:24 .
drwxrwxr-x 3 shin shin   39 May  1 14:20 ..
-rw-rw-r-- 1 shin shin 832K May  1 14:24 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 14:24 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.646914, max=0.646914, mean=0.646914
ptm: shape=(1,), min=0.676550, max=0.676550, mean=0.676550
iptm: shape=(1,), min=0.639506, max=0.639506, mean=0.639506
per_chain_ptm: shape=(1, 4), min=0.284238, max=0.746745, mean=0.517528
per_chain_pair_iptm: shape=(1, 4, 4), min=0.096109, max=0.746745, mean=0.419180
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=6.000000, mean=1.687500
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6395
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1080 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1080, 'iptm': 0.6395055651664734}
-	contact score 편차가 가장 적은것으로 판단하여 recycles = 7 고정.
```

## recycles = 8

```text
warnings.warn(_future_warning, FutureWarning)
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 8/8 [02:23<00:00, 17.97s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.43it/s]
Score=0.6605, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  1 13:52 .
drwxrwxr-x 3 shin shin   39 May  1 13:47 ..
-rw-rw-r-- 1 shin shin 831K May  1 13:52 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  1 13:52 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 1006
polymer_residues: 1006
total_atoms: 10275

========== 2) NPZ Contents (keys/shape/dtype) ==========
file: /home/shin/chai1_test/output/scores.model_idx_0.npz
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']
aggregate_score: shape=(1,), dtype=float32
ptm: shape=(1,), dtype=float32
iptm: shape=(1,), dtype=float32
per_chain_ptm: shape=(1, 4), dtype=float32
per_chain_pair_iptm: shape=(1, 4, 4), dtype=float32
has_inter_chain_clashes: shape=(1,), dtype=bool
chain_chain_clashes: shape=(1, 4, 4), dtype=int32

========== 3) NPZ Value Summary ==========
aggregate_score: shape=(1,), min=0.660494, max=0.660494, mean=0.660494
ptm: shape=(1,), min=0.685987, max=0.685987, mean=0.685987
iptm: shape=(1,), min=0.654121, max=0.654121, mean=0.654121
per_chain_ptm: shape=(1, 4), min=0.282181, max=0.746712, mean=0.521281
per_chain_pair_iptm: shape=(1, 4, 4), min=0.104899, max=0.746712, mean=0.428642
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=3.000000, mean=0.687500
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6541
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1070 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1070, 'iptm': 0.6541211605072021}
```

## 최종결론
* 7회 이상 부터는 유의미한 변화가 없다고 판단되어 recycle = 7로 고정
