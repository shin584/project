* **toy_chain :** MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP

### 실제 산출물 로그 분석 및 검증

Chai-1 시뮬레이션 구동 후 생성된 결과 파일의 내부 데이터를 파싱하여, 앞서 설정한 테스트 목적(37aa 단일 단백질 구조 생성)이 정상적으로 달성되었는지 검증했습니다.

#### 1) CIF 파일 파싱 결과 (`pred.model_idx_0.cif`)

* **데이터 출처:** `_struct.title 'Chai-1 predicted structure'`, `_audit_author.name 'Chai Discovery team'` 항목을 통해 Chai-1 모델을 통해 정상적으로 연산된 결과물임을 확인했습니다.
* **구조 및 서열 확인:** * `_struct_asym.details` 항목에서 단일 체인(`Chain A`)으로 구성되었음을 확인했습니다.
* `_pdbx_poly_seq_scheme` 루프 데이터를 통해 1번 잔기(MET)부터 37번 잔기(PRO)까지, 테스트 입력값으로 사용한 37aa 서열이 누락 없이 3D 구조로 폴딩(Folding)되었음을 확인했습니다.



#### 2) NPZ 파일 파싱 결과 (`scores.model_idx_0.npz`)

Python의 NumPy 라이브러리를 활용하여 결과 점수 파일(.npz)의 내부 배열 구조를 파싱한 결과는 다음과 같습니다.

```python
keys: ['aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes']

```

* **신뢰도 지표(Float32):** * `aggregate_score`, `ptm`, `iptm` 값이 32비트 부동소수점(Float32) 형태로 정상 생성되었습니다.
* 1차원 데이터(`(1,)`) 형태로 출력되어, 후처리 파이프라인에서 복잡한 변환 없이 즉각적인 스칼라(Scalar) 값 추출이 가능합니다.


* **충돌 검증 지표:** * `has_inter_chain_clashes`가 불리언(bool) 형태로 제공되어, 3D 구조상의 물리적 충돌 여부를 조건문(if)으로 간단히 필터링할 수 있는 구조임을 확인했습니다.

#### 종합 결론

서버 인프라 내에서 Chai-1 모델이 입력(단백질 서열)을 정확히 인식하였으며, 3D 좌표(CIF)와 모델 신뢰도 수치(NPZ)를 규격에 맞게 성공적으로 산출했습니다. 이 산출물은 향후 도입될 결합력 파싱 툴(PRODIGY-DNA 등)의 입력값으로 바로 연동 가능한 상태임을 검증했습니다.
<img width="940" height="518" alt="image" src="https://github.com/user-attachments/assets/23d97575-2738-41d9-a676-65e600ff3843" />
<img width="940" height="500" alt="image" src="https://github.com/user-attachments/assets/efd17136-79a0-4207-b46e-d70e1dca9c63" />
