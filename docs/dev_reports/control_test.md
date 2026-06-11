# cas9 대조군 실험
* 시뮬레이션 모델이 pam 서열 차이를 구분하는지 확인
* 의도적으로 pam 서열을 파괴한 타겟 서열(nagative) 입력

INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|Negative_EMX1_Non_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|Negative_EMX1_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 7/7 [02:05<00:00, 17.95s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.6383, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 May  2 13:39 .
drwxrwxr-x 3 shin shin   59 May  2 13:35 ..
-rw-rw-r-- 1 shin shin 832K May  2 13:39 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K May  2 13:39 scores.model_idx_0.npz
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
aggregate_score: shape=(1,), min=0.638308, max=0.638308, mean=0.638308
ptm: shape=(1,), min=0.669677, max=0.669677, mean=0.669677
iptm: shape=(1,), min=0.630466, max=0.630466, mean=0.630466
per_chain_ptm: shape=(1, 4), min=0.291615, max=0.738723, mean=0.508595
per_chain_pair_iptm: shape=(1, 4, 4), min=0.072360, max=0.738723, mean=0.400797
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=2.000000, mean=0.125000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.6305
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1141 points


## 결론
* pam을 의도적으로 파괴한 음성실험군에서 contact score와 ipTM의 변화가 관찰되지 않음.
* chai1 모델은 pam 서열 인식에 따른 생물학적인 구조 변화까지 예측하지는 못함
