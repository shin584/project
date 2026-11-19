"""non-NT 계열의 화면·CSV 연결을 **실제 checkpoint**로 확인한다 (#196).

계산 프로세스를 non-NT 렌더러로 띄우고, 고정 81bp 입력 하나로
  · 네 보기(사이트·Best Candidates·변종 비교·전체 상세)와 XAI의 사이트/헤드 선택,
  · 저장된 CSV의 열·순서와 끝에 더한 모델·버전,
  · 이 계열의 결과만 저장되는지,
  · non-NT 경로에서 NT 가중치·토크나이저를 준비하지 않는지
를 본다. 설명 계산은 #199/#200이므로 여기서 성공을 요구하지 않는다.

이 파일은 실제 모델을 쓰므로 가벼운 회귀와 따로 돌린다.
"""

import csv
import io
import json
import subprocess
import sys
import threading
import time
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from desktop.api import DesktopApi
from desktop.computation import ProcessDispatcher
from desktop.detail_view import BASE_COLUMNS, DETAIL_COLUMNS, MODEL_COLUMNS

FACTORY = "desktop.worker_renderer:NonNTComputationRenderer"
CHECKPOINT = ROOT / "models/non_nt/spcas9-9hydra-v2/model.keras"


@pytest.fixture(scope="module")
def dna():
    meta = json.loads((ROOT / "docs/equivalence-baseline/capture.json").read_text(encoding="utf-8"))
    return meta["input_seq"]


class Options(HTMLParser):
    """XAI 사이트 선택 `<option>`의 site_key → 헤드 목록."""

    def __init__(self, fragment):
        super().__init__()
        self.options = {}
        self.feed(fragment)

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == "option":
            self.options[data["value"]] = json.loads(data["data-heads"])


def run_requests(saved, calls):
    """요청들을 차례로 보내고 (이벤트 목록, 결과 조각들)을 돌려준다."""
    events = []
    dispatcher = ProcessDispatcher(events.append, factory_name=FACTORY)
    api = DesktopApi(dispatcher.renderer, dispatcher, lambda _: str(saved))
    try:
        for call in calls:
            call(api)
            assert dispatcher.wait_idle(600), "계산 프로세스가 끝나지 않았다"
        assert dispatcher.pid is not None, "계산 자식 프로세스가 생기지 않았다"
    finally:
        dispatcher.close()
    assert not [e for e in events if e["type"] == "error"], \
        [e["html"] for e in events if e["type"] == "error"]
    fragments = {}
    for event in events:
        if event["type"] == "result":
            fragments.update(event["fragments"])
    return events, fragments


@pytest.fixture(scope="module")
def run(tmp_path_factory, dna):
    if not CHECKPOINT.is_file():
        pytest.skip("채택 non-NT checkpoint가 없다")
    saved = tmp_path_factory.mktemp("non-nt-csv") / "crispr_prediction_results.csv"
    events, fragments = run_requests(saved, [
        lambda api: api.prepare(),
        lambda api: api.scan_sites(dna),
        lambda api: api.predict(dna),
        lambda api: api.save_csv(dna),
    ])
    return events, fragments, saved


def csv_rows(path: Path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode("utf-8-sig"), newline="")))


# ── 화면 연결 ─────────────────────────────────────────────────

def test_the_four_views_arrive_with_non_nt_predictions(run):
    _, fragments, _ = run
    assert set(fragments) >= {"sites", "best", "variants", "detail"}
    assert "후보 사이트" in fragments["sites"]
    # 예측 행이 실제로 있어야 '예측 결과가 없습니다'가 아니다.
    assert '<p class="empty">예측 결과가 없습니다.</p>' not in fragments["best"]
    assert "9-Hydra v2" in fragments["best"], "원출력 단위 안내가 이 계열의 모델을 가리켜야 한다"
    assert "SpCas9 PAM 사이트가 발견되지 않았습니다" not in fragments["variants"]


def test_xai_site_and_head_selection_lists_this_family_rows(run):
    """XAI의 사이트·헤드 선택이 이 계열의 예측 행을 가리킨다 (계산은 #199)."""
    _, fragments, _ = run
    options = Options(fragments["detail"]).options
    assert options, "XAI 사이트 선택 목록이 비어 있다"
    spcas9 = {key: heads for key, heads in options.items() if key.startswith("SpCas9|")}
    assert spcas9, "SpCas9 사이트가 선택 목록에 없다"
    assert any(heads for heads in spcas9.values()), "어느 SpCas9 사이트에도 헤드가 없다"

    # 선택 목록의 헤드는 상세 표의 예측 행 이름과 같은 집합이어야 한다.
    rows = DetailRows(fragments["detail"])
    for key, heads in spcas9.items():
        assert sorted(heads) == sorted(rows.by_site.get(key, []))


class DetailRows(HTMLParser):
    """상세 표 각 행의 site_key → 예측 행 이름."""

    def __init__(self, fragment):
        super().__init__()
        self.by_site: dict[str, list[str]] = {}
        self.feed(fragment)

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == "tr" and "data-site-key" in data:
            self.by_site.setdefault(data["data-site-key"], []).append(data["data-head"])


# ── CSV 연결 ──────────────────────────────────────────────────

def test_csv_keeps_the_existing_columns_and_appends_model_and_version(run):
    _, _, saved = run
    rows = csv_rows(saved)
    assert rows[0] == DETAIL_COLUMNS
    assert rows[0][:len(BASE_COLUMNS)] == BASE_COLUMNS
    assert rows[0][len(BASE_COLUMNS):] == MODEL_COLUMNS
    assert len(rows) > 1, "저장된 CSV에 예측 행이 없다"
    assert all(len(row) == len(DETAIL_COLUMNS) for row in rows)


def test_csv_ranks_and_sort_order_follow_the_existing_meaning(run):
    """순위는 1부터, 정렬은 기존대로 표시 점수 내림차순이다."""
    _, _, saved = run
    rows = csv_rows(saved)[1:]
    rank = DETAIL_COLUMNS.index("순위")
    score = DETAIL_COLUMNS.index("예측 점수 (%)")
    assert [int(row[rank]) for row in rows] == list(range(1, len(rows) + 1))
    scores = [float(row[score]) for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_csv_saves_only_this_family_and_names_its_checkpoint(run):
    """저장된 표의 모든 행이 현재 계열의 모델·checkpoint를 가리킨다 (#196)."""
    import model_source

    _, _, saved = run
    model = DETAIL_COLUMNS.index("모델")
    version = DETAIL_COLUMNS.index("모델 버전")
    variant = DETAIL_COLUMNS.index("Cas 변종")
    pinned = model_source.NON_NT_FILES["models/non_nt/spcas9-9hydra-v2/model.keras"]["sha256"]
    expected = f"spcas9-9hydra-v2 · sha256 {pinned[:12]}"

    spcas9 = [row for row in csv_rows(saved)[1:] if row[variant] != "Cas12a"]
    assert spcas9, "SpCas9 예측 행이 저장되지 않았다"
    assert {row[model] for row in spcas9} == {"9-Hydra v2"}
    assert {row[version] for row in spcas9} == {expected}
    # NT 계열의 모델 이름이 한 줄도 섞이지 않는다.
    assert not [row for row in csv_rows(saved)[1:] if row[model].startswith("NT ")]


def test_csv_scores_match_the_screen_rows(run):
    """CSV의 값이 화면이 보여준 예측 행과 같은 계산에서 나온다."""
    _, fragments, saved = run
    rows = csv_rows(saved)[1:]
    raw = DETAIL_COLUMNS.index("원출력")
    variant = DETAIL_COLUMNS.index("Cas 변종")
    screen = ScreenScores(fragments["detail"])
    assert screen.values, "상세 조각에 예측 행이 없다"
    columns = [variant, DETAIL_COLUMNS.index("PAM"),
               DETAIL_COLUMNS.index("가이드 서열 (RNA)"), DETAIL_COLUMNS.index("가닥"),
               DETAIL_COLUMNS.index("거리 (bp)")]
    for row in rows:
        key = tuple(row[index] for index in columns)
        assert key in screen.values
        assert float(row[raw]) == pytest.approx(round(screen.values[key], 4), abs=5e-5)


class ScreenScores(HTMLParser):
    """상세 조각의 (예측 행 이름, PAM, 가이드, 가닥, 거리) → 반올림 전 원출력.

    한 변종이 여러 사이트를 인식하므로 이름·PAM·가닥만으로는 행이 가려지지 않는다.
    """

    def __init__(self, fragment):
        super().__init__()
        self.values: dict[tuple[str, str, str], float] = {}
        self._pending = None
        self._cells: list[str] = []
        self._cell = None
        self.feed(fragment)

    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag == "tr" and "data-raw" in data:
            self._pending = (data["data-cas-name"], float(data["data-raw"]))
            self._cells = []
        elif tag == "td":
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self._cell is not None:
            self._cells.append("".join(self._cell))
            self._cell = None
        elif tag == "tr" and self._pending is not None:
            name, value = self._pending
            self.values[(name, self._cells[2], self._cells[3],
                         self._cells[4], self._cells[5])] = value
            self._pending = None


# ── NT를 건드리지 않는다 ──────────────────────────────────────

def test_the_non_nt_path_loads_no_nt_weights_or_tokenizer(dna):
    """이 계열을 쓰는 프로세스는 NT 런타임을 준비하지도, 데우지도 않는다 (#196)."""
    if not CHECKPOINT.is_file():
        pytest.skip("채택 non-NT checkpoint가 없다")
    script = f"""
import sys
import core, predict
from desktop.worker_renderer import NonNTComputationRenderer

renderer = NonNTComputationRenderer()
fragments = renderer.prediction_fragments({dna!r}, lambda done, total: None)
assert fragments["best"], "no prediction fragment"

rows = renderer.csv_table({dna!r}, lambda done, total: None)
assert len(rows) > 1, "no csv rows"

# NT 토크나이저는 한 번도 불리지 않았다.
assert predict.load_nt_tokenizer.cache_info().currsize == 0, predict.load_nt_tokenizer.cache_info()
# NT 가중치 형식을 읽는 라이브러리가 아예 올라오지 않았다.
leaked = {{"transformers", "safetensors"}} & set(sys.modules)
assert not leaked, leaked
# 이 계열의 예측기만 만들어졌다.
predictors = core.load_predictors(warmup=False, family="non_nt")
assert not any(type(p).__name__ == "NTClassifierPredictor" for p in predictors.values())
print("ok")
"""
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=900, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_a_sequence_without_spcas9_sites_loads_no_checkpoint():
    """해당 Cas의 사이트가 없으면 그 예측기는 메모리에 올라가지 않는다."""
    script = """
import core
import non_nt_predict as nnp

predictors = core.load_predictors(warmup=False, family="non_nt")
spcas9 = predictors[("A", "SpCas9")]
assert type(spcas9).__name__ == "NonNTSpCas9Predictor"
assert spcas9.predict_batch([]) == []
assert spcas9._runtime is None
import sys
assert "keras" not in sys.modules, "예측하지 않았는데 Keras가 올라왔다"
print("ok")
"""
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout
