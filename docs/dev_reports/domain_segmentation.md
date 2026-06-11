# Tier 0 spCas9 전체 서열
-	코드 스니펫
-	>protein|name=SpCas9
-	MDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAE
-	ATRLKRTARRRYTRRKNRICYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFG
-	NIVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAHMIKFRGHFLIEGDLNPDNSD
-	VDKLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGLFGN
-	LIALSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAI
-	LLSDILRVNTEITKAPLSASMIKRYDEHHQDLTLLKALVRQQLPEKYKEIFFDQSKNGYA
-	GYIDGGASQEEFYKFIKPILEKMDGTEELLVKLNREDLLRKQRTFDNGSIPHQIHLGELH
-	AILRRQEDFYPFLKDNREKIEKILTFRIPYYVGPLARGNSRFAWMTRKSEETITPWNFEE
-	VVDKGASAQSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAFL
-	SGEQKKAIVDLLFKTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLKI
-	IKDKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWG
-	RLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSL
-	HEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRER
-	MKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDH
-	IVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNL
-	TKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKS
-	KLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRK
-	MIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDF
-	ATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVA
-	YSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPK
-	YSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVE
-	QHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGA
-	PAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD
-	>dna|name=Target_DNA
-	ATGCGTACGTAGCTAGCTAGNNGRRT

##결과	
vram 초과 발생
---

# tier 1 PI 도메인(약 230aa)
##결과
* contact score : 812
*	ipTM : 0.0974
*	점수가 저조할 뿐만 아니라 편차도 매우 큼.


# Tier 2 (WED + PI + sgRNA + dsDNA)
```fasta
>protein|SpCas9_WED_PI IKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD >rna|EMX1_sgRNA GAGUCCGAGCAGAAGAAGAAGUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUU >dna|EMX1_Target_Strand TCAGCCCTTCTTCTTCTGCTCGGACTCGGC >dna|EMX1_Non_Target_Strand GCCGAGTCCGAGCAGAAGAAGAAGGGCTGA

========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 530
polymer_residues: 530
total_atoms: 6366

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
aggregate_score: shape=(1,), min=0.336646, max=0.336646, mean=0.336646
ptm: shape=(1,), min=0.426933, max=0.426933, mean=0.426933
iptm: shape=(1,), min=0.314075, max=0.314075, mean=0.314075
per_chain_ptm: shape=(1, 4), min=0.209862, max=0.758456, mean=0.409790
per_chain_pair_iptm: shape=(1, 4, 4), min=0.009224, max=0.758456, mean=0.276321
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=6.000000, mean=1.375000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.3141
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1409 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1409, 'iptm': 0.3140745162963867}
```

*	**2차 시기**
```fasta
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 530
polymer_residues: 530
total_atoms: 6366

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
aggregate_score: shape=(1,), min=0.338157, max=0.338157, mean=0.338157
ptm: shape=(1,), min=0.423794, max=0.423794, mean=0.423794
iptm: shape=(1,), min=0.316748, max=0.316748, mean=0.316748
per_chain_ptm: shape=(1, 4), min=0.214340, max=0.758228, mean=0.412574
per_chain_pair_iptm: shape=(1, 4, 4), min=0.010745, max=0.758228, mean=0.278852
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=2.000000, mean=0.562500
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.3167
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 996 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 996, 'iptm': 0.3167475759983063}
```
## 결과
여전히 분산이 큼
---
# Tier 3 RuvC II & III, HNH, WED, PI 도메인
```fasta
>protein|SpCas9_Tier2_NUC DKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWGRLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSLHEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKSKLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD >rna|EMX1_sgRNA GAGUCCGAGCAGAAGAAGAAGUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUU >dna|EMX1_Target_Strand TCAGCCCTTCTTCTTCTGCTCGGACTCGGC >dna|EMX1_Non_Target_Strand GCCGAGTCCGAGCAGAAGAAGAAGGGCTGA

-	1차 시기
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
aggregate_score: shape=(1,), min=0.535949, max=0.535949, mean=0.535949
ptm: shape=(1,), min=0.574398, max=0.574398, mean=0.574398
iptm: shape=(1,), min=0.526336, max=0.526336, mean=0.526336
per_chain_ptm: shape=(1, 4), min=0.266711, max=0.743087, mean=0.467501
per_chain_pair_iptm: shape=(1, 4, 4), min=0.073938, max=0.743087, mean=0.349710
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=5.000000, mean=1.000000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.5263
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1461 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1461, 'iptm': 0.526336133480072}
```
* **결과:** contact score의 큰 상승은 없지만, iptm의 유의미한 상승이 관찰됨

* **2차 시기**
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 925
polymer_residues: 925
total_atoms: 9615

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
aggregate_score: shape=(1,), min=0.517383, max=0.517383, mean=0.517383
ptm: shape=(1,), min=0.557425, max=0.557425, mean=0.557425
iptm: shape=(1,), min=0.507372, max=0.507372, mean=0.507372
per_chain_ptm: shape=(1, 4), min=0.264580, max=0.743560, mean=0.462832
per_chain_pair_iptm: shape=(1, 4, 4), min=0.080980, max=0.743560, mean=0.350807
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=6.000000, mean=1.250000
(chai1) shin@TeamB511105:~$ ./parsin.sh
bash: ./parsin.sh: No such file or directory
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.5074
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1164 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1164, 'iptm': 0.5073719024658203}

* **3차 시기**
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_Tier2_NUC 766
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 3/3 [00:53<00:00, 17.80s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.45it/s]
Score=0.5219, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 780K
drwxrwxr-x 2 shin shin   64 Apr 30 13:55 .
drwxrwxr-x 3 shin shin   39 Apr 30 13:52 ..
-rw-rw-r-- 1 shin shin 776K Apr 30 13:55 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K Apr 30 13:55 scores.model_idx_0.npz
(chai1) shin@TeamB511105:~$ ./summary.sh
========== 1) CIF Summary ==========
file: /home/shin/chai1_test/output/pred.model_idx_0.cif
models: 1
chains: 4
chain_names: ['A', 'B', 'C', 'D']
total_residues: 925
polymer_residues: 925
total_atoms: 9615

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
aggregate_score: shape=(1,), min=0.521932, max=0.521932, mean=0.521932
ptm: shape=(1,), min=0.561621, max=0.561621, mean=0.561621
iptm: shape=(1,), min=0.512009, max=0.512009, mean=0.512009
per_chain_ptm: shape=(1, 4), min=0.265459, max=0.744571, mean=0.464622
per_chain_pair_iptm: shape=(1, 4, 4), min=0.074300, max=0.744571, mean=0.349368
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=10.000000, mean=1.687500
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.5120
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1095 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1095, 'iptm': 0.5120094418525696}


## 결과
contact score의 편차는 여전히 존재하나 iptm은 비교적 안정된 모습을 보임.


# tier3.5 REC3 + NUC + sgRNA + dsDNA
```fasta
>protein|SpCas9_Tier2.5_REC3_NUC AWMTRKSEETITPWNFEEVVDKGASAQSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAFLSGEQKKAIVDLLFKTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLKIIKDKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWGRLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSLHEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKSKLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD >rna|EMX1_sgRNA GAGUCCGAGCAGAAGAAGAAGUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUU >dna|EMX1_Target_Strand TCAGCCCTTCTTCTTCTGCTCGGACTCGGC >dna|EMX1_Non_Target_Strand GCCGAGTCCGAGCAGAAGAAGAAGGGCTGA

RuntimeError: CUDA out of memory. Tried to allocate 4.50 GiB. GPU 0 has a total capacity of 23.52 GiB of which 3.15 GiB is free. 
Process 296138 has 20.36 GiB memory in use. Of the allocated memory 16.25 GiB is allocated by PyTorch, and 3.65 GiB is reserved by
PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
to avoid fragmentation.  See documentation for Memory Management 
```
## 결과
vram 초과 발생

# tier 3.3 
```fasta
>protein|SpCas9_True_Tier2.3_REC3_Half
NELTKVKYVTEGMRKPAFLSGEQKKAIVDLLFKTNRKVTVKQLKEDYFKKIECFDSVEISGVEDRFNASLGTYHDLLKIIKDKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLKRRRYTGWGRLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSLHEHIANLAGSPAIKKGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYLQNGRDMYVDQELDINRLSDYDVDHIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFDNLTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKSKLVSDFRKDFQFYKVREINNYHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPLIETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVLVVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNELALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAENIIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD
>rna|EMX1_sgRNA
GAGUCCGAGCAGAAGAAGAAGUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGCUUU
>dna|EMX1_Target_Strand
TCAGCCCTTCTTCTTCTGCTCGGACTCGGC
>dna|EMX1_Non_Target_Strand
GCCGAGTCCGAGCAGAAGAAGAAGGGCTGA
-	1차 시기
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 3/3 [00:53<00:00, 17.82s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.5918, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 Apr 30 14:02 .
drwxrwxr-x 3 shin shin   39 Apr 30 13:59 ..
-rw-rw-r-- 1 shin shin 831K Apr 30 14:02 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K Apr 30 14:02 scores.model_idx_0.npz
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
aggregate_score: shape=(1,), min=0.591825, max=0.591825, mean=0.591825
ptm: shape=(1,), min=0.639732, max=0.639732, mean=0.639732
iptm: shape=(1,), min=0.579849, max=0.579849, mean=0.579849
per_chain_ptm: shape=(1, 4), min=0.264102, max=0.740239, mean=0.487660
per_chain_pair_iptm: shape=(1, 4, 4), min=0.081366, max=0.740239, mean=0.364974
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=10.000000, mean=3.125000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.5798
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1909 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1909, 'iptm': 0.5798485279083252}
```

*	**2차시기**

```fasta
9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 3/3 [00:53<00:00, 17.82s/it] 
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s] 
Score=0.5918, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 836K
drwxrwxr-x 2 shin shin   64 Apr 30 14:02 .
drwxrwxr-x 3 shin shin   39 Apr 30 13:59 ..
-rw-rw-r-- 1 shin shin 831K Apr 30 14:02 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K Apr 30 14:02 scores.model_idx_0.npz
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
aggregate_score: shape=(1,), min=0.591825, max=0.591825, mean=0.591825
ptm: shape=(1,), min=0.639732, max=0.639732, mean=0.639732
iptm: shape=(1,), min=0.579849, max=0.579849, mean=0.579849
per_chain_ptm: shape=(1, 4), min=0.264102, max=0.740239, mean=0.487660
per_chain_pair_iptm: shape=(1, 4, 4), min=0.081366, max=0.740239, mean=0.364974
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=10.000000, mean=3.125000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.5798
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1909 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1909, 'iptm': 0.5798485279083252}

* **3차시기**

```
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] protein|SpCas9_True_Tier2.3_REC3_Half 847
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] rna|EMX1_sgRNA 99
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Target_Strand 30
INFO:chai_lab.data.dataset.inference_dataset:[fasta] [/home/shin/chai1_test/input.fasta] dna|EMX1_Non_Target_Strand 30
INFO:root:Trunk sample 1/1
Trunk recycles: 100%|███████████████████████████████████████████████████████████████████████████████| 3/3 [00:53<00:00, 17.83s/it]
Diffusion steps: 100%|██████████████████████████████████████████████████████████████████████████| 199/199 [01:21<00:00,  2.44it/s]
Score=0.5875, writing output to /home/shin/chai1_test/output/pred.model_idx_0.cif
INFO:chai_lab.data.io.cif_utils:saved cif file to /home/shin/chai1_test/output/pred.model_idx_0.cif
total 840K
drwxrwxr-x 2 shin shin   64 Apr 30 14:15 .
drwxrwxr-x 3 shin shin   39 Apr 30 14:12 ..
-rw-rw-r-- 1 shin shin 833K Apr 30 14:15 pred.model_idx_0.cif
-rw-rw-r-- 1 shin shin 2.0K Apr 30 14:15 scores.model_idx_0.npz
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
aggregate_score: shape=(1,), min=0.587550, max=0.587550, mean=0.587550
ptm: shape=(1,), min=0.625481, max=0.625481, mean=0.625481
iptm: shape=(1,), min=0.578067, max=0.578067, mean=0.578067
per_chain_ptm: shape=(1, 4), min=0.272642, max=0.740507, mean=0.484845
per_chain_pair_iptm: shape=(1, 4, 4), min=0.062444, max=0.740507, mean=0.355433
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=8.000000, mean=2.187500
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.5781
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1599 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1599, 'iptm': 0.5780668258666992}
```

## 결론
5.5개의 도메인(RuvC II/III, HNH, WED, PI, REC3 Half) 구성된 약 730aa 서열 최종 채택
