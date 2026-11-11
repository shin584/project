# NT XAI 데모 자료 (`data/`)

`python -m xai_test.nt_xai_demo prepare`가 두 뉴클레아제의 저장 산출물을 이
디렉터리 하나에 모으고 `data/manifest.json` 1개로 체크섬한다. 미팅 직전 온전성
확인은 `python -m xai_test.nt_xai_demo verify` 한 번이다.

## 복원 조건: **Cas12a ZIP + SpCas9 원본 묶음**

Git에는 경량 JSON·SVG·`cases.csv`와 `manifest.json`만 있다. `.npz`와 per-row CSV
(manifest의 `"git": "ignored"` 항목)는 Git 밖이며, 아래 두 원본이 제자리에
있을 때 `prepare`로 그대로 다시 만들어진다. 원본이 고정해 둔 해시와 대조한 **뒤에만**
쓰므로, 원본이 바뀌었으면 manifest를 만들지 않고 실패한다.

### 1. Cas12a ZIP

| 로컬 경로 | sha256 |
|---|---|
| `xai_test/cas12a_NT_analysis/result/cas12a-test1500-results-a9e50fa7f668.zip` (58,644,763 B) | `40cbd2f24f0c19bec42b776282b407a5d9538129d6b5543dcc2c3c29d3ab098f` |

| `xai_test/cas12a_NT_analysis/result/test1500-a9e50fa7f668-final-audit.json` (1,699 B) | ZIP의 고정 해시 출처 (`archive_sha256`, `status: PASS`) |

**두 파일 모두 필요하며 둘 다 Git 밖이다** (`result/`는 추적되지 않는다). 감사 파일이
없으면 `prepare`는 ZIP 해시를 대조할 수 없어 실패한다. ZIP은 Colab `/content/`에서
만들어져 내려받은 것이고 저장소에 Drive 사본 위치 기록은 없다. 또 `archive_sha256`은
반환 **후** 계산된 값이라 별도 채널의 trusted checksum이 아니다: 반환 이후의 변경은
잡지만 반환 전의 손상은 잡지 못한다.

꺼내는 것: `plan.json`, `plan.sha256`, `execution.json`, `cases.json`, 그리고
`reports/745133f240e1/`의 경량 파일(`predictions.csv`, `metrics.json`, `summary.json`,
`position_profiles.json`, `token_profiles.json`, `rollout_summary.json`,
`cases_annotated.json`, `cases.csv`, SVG 3개, `report.ko.md`)과 `ism.npz` / `ig.npz` /
`rollout.npz`.

꺼내지 않는 것: `shards/`(per-row 10,500개), `sessions/`, 모든 `*.sidecar.json`
(ism 47MB, rollout 12MB, ig 8.6MB, predictions 0.4MB), per-row `rollout_diagnostics.json`
(5MB). sidecar는 ZIP 안에서 읽어 `npz_sha256`·`plan_sha256` 대조에만 쓴다.

`agreement.json`(32MB)은 `agreement.sliced.json`으로 **슬라이스**한다: 최상위 키
`populations` / `group_diagnostics` / `protocol`만 원본 값 그대로 남기고 `rows`(1,500행),
`statistics_ids`, `case_only_ids`, `ig_mode`를 버린다. 산술은 없다. 원본 sha256은
manifest의 `derived`에 기록된다.

### 2. SpCas9 원본 묶음 (9-Hydra v3)

모두 현재 Git 추적 대상이 아니다 (`.gitignore`의 `*.csv`와
`xai_test/SpCas9_NT_analysis/artifacts/**/*.npz`). 묶음 중 하나라도 없으면 `prepare`는
Cas12a만 준비하고, manifest의 `nucleases.spcas9`에 `"status": "absent"`와 없는 파일
목록을 쓰며, 콘솔에 SpCas9 부재를 출력한다.

| 배치할 로컬 경로 | 바이트 | sha256 | Drive 위치 |
|---|---:|---|---|
| `xai_test/SpCas9_NT_analysis/artifacts/v3_ism/ism.npz` | 16,922,326 | `9cc7b78c9c280a427a560e37ef3096f171cd38ee322550f37902d74ce11abe5b` | `MyDrive/SpCas9/spcas9_v3/artifacts/v3_ism/ism.npz` |
| `xai_test/SpCas9_NT_analysis/artifacts/v3_ig/ig.npz` | 47,470 | `e7973fa58fb782c1c9ec6310f3416a12c122e6a27865444608fbe02d53c063f2` | `MyDrive/SpCas9/spcas9_v3/artifacts/v3_ig/ig.npz` |
| `xai_test/SpCas9_NT_analysis/artifacts/v3_rollout/rollout.npz` | 416,426 | `b42b7b6dbb89db6b1a8a16b631ba62d196d86437799aa38dd0a95a784097c6b4` | `MyDrive/SpCas9/spcas9_v3/artifacts/v3_rollout/rollout.npz` |
| `xai_test/SpCas9/published_split/test.csv` | 1,206,283 | `1a1305b6e0eeef91625443f1e21f85d233f8e0dac9e8be3cdb3daea5eeaf549b` | `MyDrive/SpCas9/test.csv` |
| `xai_test/SpCas9/published_split/train.csv` | 6,915,393 | `4c1ffc29c4db87cab5e5da1d29abe3a84440050c66b1a6c3b30daaf723812c01` | Drive 경로 기록 없음 |
| `xai_test/SpCas9/published_split/val.csv` | 873,210 | `0757d7c2f7766e13d4c9a4d62fd1b6570ca17076087b711c139ec1ff6b54728d` | Drive 경로 기록 없음 |
| `xai_test/SpCas9_NT_analysis/artifacts/v3_testset_predictions/predictions.csv` | 1,122,862 | `7b4f9b16c048a2d0ba9abb9699b32a3e181954c6ba35ce7f002cea6f9850974a` | Drive 경로 기록 없음 (로컬 CPU 실행) |

Drive 위치는 `SpCas9_v3.ipynb`의 `OUT_DIR`/`SPLIT_CSV` 설정과 각 sidecar의
`split.path`에서 온 것이다. "기록 없음" 세 파일은 Drive 사본의 위치를 저장소가 적어 둔
적이 없다. 다시 만들 수는 있다: `train.csv`/`val.csv`는 `split_checksum.json`의
seed 42 분할, `predictions.csv`는 `v3_predict.export_predictions`(고정 checkpoint +
`test.csv`)로 재생성되며 위 sha256이 같아야 한다.

`prepare`가 대조하는 SpCas9 고정 해시: 각 sidecar의 `delta_sha256` /
`attribution_sha256` / `rollout_sha256`과 `population_digest` / `membership_digest`
(npz의 `row_index`로 재계산), `split_checksum.json`, `predictions_checksum.json`,
`*_checksum.json` 3개, 그리고 `cases_annotated.json` / `head_agreement.json` /
`ig_ism_validation.json` / `head_order_reversal.json`의 `consumed` digest.

## 테스트

`python -m pytest xai_test/nt_xai_demo/test -q`. 합성 fixture 테스트는 어디서나 돈다.
`test_real_data_matches_its_manifest`는 실제 bulk 자료가 있을 때만 돌고 없으면
skip된다. 이 테스트는 파일 변경 탐지일 뿐이며 리더 오독을 잡지 않는다.
