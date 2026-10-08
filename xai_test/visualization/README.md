# Model B 설명 대시보드

SaCas9 Model B 예측의 근거(ISM, Attribution Logo, Attention Rollout, Token-grouped SHAP)를
Testset 514개 샘플에 대해 보여주는 Streamlit 앱입니다. Colab에서 사전 계산한 결과만
읽으므로 GPU나 NT 모델 가중치가 필요 없습니다. 계획은 `plan_draft.md`를 참고하세요.

## 실행

```bash
git clone -b visualization https://github.com/shin584/project.git
cd project/xai_test/visualization
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 http://localhost:8501 이 열립니다. 왼쪽 사이드바에서 `sample_id`를 고르세요
("IG 있음" 표시 샘플은 Attribution Logo까지 볼 수 있습니다).

## 필요한 데이터 (저장소에 포함됨)

- `xai_test/test_metadata.csv`: Testset 서열과 실측값
- `xai_test/final_analysis_result/`: `model_analysis_arrays.npz`, `model_analysis_summary.json`,
  `model_b_testset_predictions.csv`

## 테스트

```bash
cd xai_test/visualization
python -m pytest test_data_source.py test_app.py -q
```

`test_data_source.py`의 구간 경계 검사는 상위 폴더의 분석 모듈을 import하므로 `torch` 등이
추가로 필요합니다. 시연 전 수동 점검은 `demo_checklist.md`를 따르세요.
