# cas12a simulation result by Chai-1

## Cas 단백질 도메인 구성 : N말단 WED / C말단 WED / PI도메인 / RuvC,Nuc 엽

'''fasta
>protein|AsCas12a_WED_Fusion_837aa
MTQFEGFTNLYQVSKTLRFELIPQGKTLKHIQEQGFIEEDKARNDHYKELKPIIDRIYKTYADQCLQLVQLDWWDVNKEK
NNGAILFVKNGLYYLGIMPKQKGRYKALSFEPTEKTSEGFDKMYYDYFPDAAKMIPKCSTQLKAVTAHFQTHTTPILLSN
NFIEPLEITKEIYDLNNPEKEPKKFQTAYAKKTGDQKGYREALCKWIDFTRDFLSKYTKTTSIDLSSLRPSSQYKDLGEY
YAELNPLLYHISFQRIAEKEIMDAVETGKLYLFQIYNKDFAKGHHGKPNLHTLYWTGLFSPENLAKTSIKLNGQAELFYR
PKSRMKRMAHRLGEKMLNKKLKDQKTPIPDTLYQELYDYVNHRLSHDLSDEARALLPNVITKEVSHEIIKDRRFTSDKFF
FHVPITLNYQAANSPSKFNQRVNAYLKEHPETPIIGIDRGERNLIYITVIDSTGKILEQRSLNTIQQFDYQKKLDNREKE
RVAARQAWSVVGTIKDLKQGYLSQVIHEIVDLMIHYQAVVVLENLNFGFKSKRTGIAEKAVYQQFEKMLIDKLNCLVLKD
YPAEKVGGVLNPYQLTDQFTSFAKMGTQSGFLFYVPAPYTSKIDPLTGFVDPFVWKTIKNHESRKHFLEGFDFLHYDVKT
GDFILHFKMNRNLSFQRGLPGFMPAWDIVFEKNETQFDAKGTPFIAGKRIVPVIENHRFTGRYRDLYPANELIALLEEKG
IVFRDGSNILPKLLENDDSHAIDTMVALIRSVLQMRNSNAATGEDYINSPVRDLNGVCFDSRFQNPEWPMDADANGAYHI
ALKGQLLLNHLKESKDLKLQNGISNQDWLAYIQELRN
>rna|AsCas12a_crRNA
UAAUUUCUACUCUUGUAGAUCUGGGGCAGGGACUCCACCA
>dna|Target_DNA_Non_Target_Strand
TTGATTTCCTGGGGCAGGGACTCCACCAGCTTCT
>dna|Target_DNA_Target_Strand
AGAAGCTGGTGGAGTCCCTGCCCCAGGAAATCAA
'''

---
## 테스트 결과

'''
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
aggregate_score: shape=(1,), min=0.548834, max=0.548834, mean=0.548834
ptm: shape=(1,), min=0.763001, max=0.763001, mean=0.763001
iptm: shape=(1,), min=0.495292, max=0.495292, mean=0.495292
per_chain_ptm: shape=(1, 4), min=0.385453, max=0.800436, mean=0.536778
per_chain_pair_iptm: shape=(1, 4, 4), min=0.025005, max=0.800436, mean=0.263495
has_inter_chain_clashes: shape=(1,), min=0.000000, max=0.000000, mean=0.000000
chain_chain_clashes: shape=(1, 4, 4), min=0.000000, max=8.000000, mean=1.000000
(chai1) shin@TeamB511105:~$ ./parsing.sh
[INFO] Chai-1 ipTM (신뢰도) 추출 완료: 0.4953
[INFO] 물리적 접촉 점수(Contact Score) 연산 완료: 1306 points

[SUCCESS] 최종 추출된 1차원 피처 데이터:
{'contact_score': 1306, 'iptm': 0.49529239535331726}

'''
