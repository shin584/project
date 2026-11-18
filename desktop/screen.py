"""Self-contained shell, supplied to pywebview as HTML rather than a URL.

The six fragment slots and onEvent entry point follow docs/desktop-boundary.md.
No asset requests, CDN, local HTTP server, or model imports are needed.
"""

SCREEN_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CRISPR PAM Scanner + XAI</title>
  <style>
    :root { font-family: "Segoe UI", "Malgun Gothic", sans-serif; color: #202938;
      background: #fff; font-size: 15px; }
    * { box-sizing: border-box; }
    body { margin: 0; display: flex; min-height: 100vh; }
    aside { width: 280px; flex-shrink: 0; background: #f4f6f9; padding: 30px 24px;
      border-right: 1px solid #e2e7ee; }
    main { flex: 1; min-width: 0; padding: 30px 36px; }
    h1 { font-size: 28px; margin: 0 0 12px; }
    h2 { font-size: 20px; margin: 0 0 18px; }
    p { line-height: 1.65; }
    label { display: block; margin: 20px 0 8px; }
    input[type=text] { width: 100%; padding: 11px; border: 1px solid #bbc6d4;
      border-radius: 6px; font: inherit; }
    input[type=range] { width: 100%; accent-color: #245fa6; }
    button { border: 1px solid #bbc6d4; border-radius: 6px; background: white;
      padding: 10px 14px; font: inherit; cursor: pointer; }
    button:focus-visible, input:focus-visible { outline: 3px solid #8cb6ec; outline-offset: 2px; }
    button:disabled { opacity: .5; cursor: default; }
    #analyze { display: block; width: 100%; margin-top: 16px; color: white;
      border-color: #245fa6; background: #245fa6; }
    #model-status { min-height: 48px; margin: 20px 0; }
    #model-status p { margin: 0; }
    .caption { font-size: 13px; color: #58677c; }
    .metrics { display: flex; flex-wrap: wrap; gap: 30px; margin: 20px 0; }
    .metrics span, .metrics strong { display: block; }
    .metrics strong { font-size: 24px; margin-top: 8px; }
    .success { padding: 16px; background: #e6f4ea; border-radius: 6px; }
    .demo, .warning, #notice:not(:empty) { padding: 12px 16px; background: #fff6df;
      border: 1px solid #ecd7a2; border-radius: 6px; }
    .error { color: #ac2830; }
    #welcome { padding: 14px 18px; background: #edf4fc; border-radius: 6px; }
    [role=tablist] { display: flex; flex-wrap: wrap; gap: 8px; margin: 26px 0 22px;
      border-bottom: 1px solid #dce3eb; padding-bottom: 12px; }
    [role=tab][aria-selected=true] { color: #174b89; background: #e9f1fb;
      border-color: #245fa6; font-weight: 600; }
    [role=tabpanel] { overflow-x: auto; }
    [hidden] { display: none !important; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e1e7ef; text-align: left;
      white-space: nowrap; }
    th { background: #f0f4f8; }
    tr:hover td { background: #f7faff; }
    /* The score cell keeps its own inline gradient, as in the Streamlit Styler. */
    tr.selected td { background: #ffe8a3; color: #000; }
    .filter-columns { display: flex; flex-wrap: wrap; gap: 20px; }
    .filter-columns fieldset { flex: 1 1 320px; min-width: 0; }
    .checklist { max-height: 280px; overflow-y: auto; padding: 4px 12px;
      border: 1px solid #dce3eb; border-radius: 6px; }
    .checklist label { display: block; margin: 8px 0; }
    figure { margin: 20px 0; }
    figure.xai { margin-left: auto; margin-right: auto; }
    figure img { width: 100%; display: block; }
    select, input[type=number] { font: inherit; padding: 8px; max-width: 100%; }
    select { width: 100%; }
    fieldset { margin: 20px 0; border: 1px solid #dce3eb; }
    fieldset label { display: inline-block; margin: 10px 16px 10px 0; }
    progress { width: min(500px, 70%); }
    .window-sequence { font: 22px monospace; letter-spacing: 2px; overflow-wrap: anywhere; }
    .xai-window h3 { font-size: 16px; margin: 10px 0 6px; }
    .xai-window p { font-size: 13px; line-height: 1.4; margin: 6px 0; }
    figcaption { margin-top: 12px; color: #58677c; }
    details { margin: 18px 0; }
    summary { cursor: pointer; }
    hr { border: 0; border-top: 1px solid #dce3eb; margin: 24px 0; }
    @media (max-width: 1000px) { aside { width: 240px; padding: 24px 18px; }
      main { padding: 24px; } }
  </style>
</head>
<body>
  <aside aria-label="입력">
    <h2>서열 입력</h2>
    <form id="analysis-form">
      <label for="dna">81bp DNA 서열</label>
      <input id="dna" type="text" placeholder="ATCG..." autocomplete="off" spellcheck="false">
      <button id="analyze" type="submit" disabled>분석 시작</button>
    </form>
    <div id="model-status" role="status" aria-live="polite">모델 준비 중</div>
    <button id="retry-prepare" type="button" hidden>모델 준비 다시 시도</button>
    <hr>
    <label for="graph-size">XAI 그래프 크기</label>
    <input id="graph-size" type="range" min="40" max="100" value="80" step="5"
      aria-describedby="graph-size-help">
    <output id="graph-size-value" for="graph-size">80%</output>
    <p id="graph-size-help" class="caption">XAI 탭의 그래프 가로 폭입니다. 줄이면 그래프 높이도 함께 줄어듭니다.</p>
  </aside>
  <main>
    <h1>CRISPR PAM Scanner + XAI</h1>
    <p>81bp DNA 서열을 입력하면 PAM 탐색, 절단 효율 예측, 예측 근거(XAI)를 보여줍니다.</p>
    <p class="demo">화면 미리보기 · 예측 점수와 XAI는 가짜 값입니다. 실제 분석에 사용하지 마세요.</p>
    <p id="welcome">왼쪽 사이드바에 81bp DNA 서열을 입력하고 <strong>분석 시작</strong>을 눌러 주세요.</p>
    <div id="notice" data-slot="notice" role="alert" aria-live="polite"></div>
    <div id="task-state" role="status" aria-live="polite" hidden>
      <span id="task-label"></span>
      <progress id="task-progress" max="100"></progress>
      <button id="cancel-task" type="button">계산 취소</button>
    </div>
    <div role="tablist" aria-label="보기">
      <button id="tab-best" role="tab" aria-selected="true" aria-controls="view-best" tabindex="0">Best Candidates</button>
      <button id="tab-variants" role="tab" aria-selected="false" aria-controls="view-variants" tabindex="-1">변종 비교</button>
      <button id="tab-detail" role="tab" aria-selected="false" aria-controls="view-detail" tabindex="-1">전체 상세 보기</button>
      <button id="tab-xai" role="tab" aria-selected="false" aria-controls="view-xai" tabindex="-1">XAI 설명</button>
    </div>
    <section id="view-best" role="tabpanel" aria-labelledby="tab-best">
      <section id="results" hidden>
        <h2>PAM 탐색 결과</h2>
        <div data-slot="sites"></div>
        <p class="caption">모델 입력 윈도우가 서열 밖으로 나가는 사이트는 예측할 수 없어 제외됩니다.</p>
        <h2>절단 효율 예측 결과</h2>
        <p class="caption">점수의 의미는 Cas 종류마다 다릅니다 — Cas12a는 <strong>예측 절단 효율(%)</strong>,
          SpCas9은 <strong>학습 분포 백분위(%)</strong> 입니다. 자세한 내용은 아래
          '모델 출력 해석 상태'를 펼쳐 보세요. 거리 보정은 적용하지 않았고,
          변이와의 거리는 '거리 (bp)' 열에서 확인할 수 있습니다.</p>
      </section>
      <div data-slot="best"><p>예측 결과가 없습니다.</p></div>
    </section>
    <section id="view-variants" role="tabpanel" aria-labelledby="tab-variants" hidden>
      <div data-slot="variants"><p>예측 결과가 없습니다.</p></div>
    </section>
    <section id="view-detail" role="tabpanel" aria-labelledby="tab-detail" hidden>
      <div data-slot="detail"><p>예측 결과가 없습니다.</p></div>
    </section>
    <section id="view-xai" role="tabpanel" aria-labelledby="tab-xai" hidden>
      <h2>모델 예측 근거 상세 분석 (XAI)</h2>
      <details><summary>처음이신가요? 5분이면 되는 배경 설명</summary>
        <h3>이 도구가 하는 일</h3><p>CRISPR 단백질은 DNA를 자르는 가위입니다.
        이 도구는 자를 수 있는 지점을 찾고 절단 효율과 모델의 예측 근거를 보여줍니다.</p>
        <h3>서열의 세 구간</h3><p>PAM은 가위가 찾는 표지판, guide는 자를 위치를 지정하는 주소,
        flank는 주변 환경입니다.</p><h3>숫자 읽는 법</h3>
        <p>예측 절단 효율은 높을수록 좋습니다. %p는 효율의 변화량으로, 30%에서 35%는 +5%p입니다.
        변이와의 거리는 고치려는 지점과 절단 위치 사이의 거리이며 가까울수록 유리합니다.</p>
        <h3>세 가지 분석 방법의 차이</h3><p>Attention Rollout은 자주 참고한 자리(빠름),
        IG는 각 글자가 효율을 올렸는지 내렸는지(보통), SHAP은 평균적인 서열 대비 각 자리의 기여(느림)를 봅니다.
        서로 다른 질문에 답하므로 결과가 달라도 모순이 아닙니다.</p>
      </details>
      <form id="xai-controls" hidden>
        <label for="xai-site">분석 대상 사이트 (효율 높은 순)</label>
        <select id="xai-site"></select>
        <div id="xai-head-control">
          <label for="xai-head">설명할 출력(변종)</label>
          <select id="xai-head"></select>
        </div>
        <p id="xai-single-head" hidden></p>
        <div id="xai-window"></div>
        <div class="metrics" id="xai-metrics"></div>
        <details>
          <summary>계산 설정</summary>
          <p class="caption">IG는 단계 수만큼, SHAP은 (샘플 수 × 배경 수)만큼 추론을 반복합니다.</p>
          <label for="ig-steps">IG 적분 단계 수</label>
          <input id="ig-steps" type="number" min="8" max="256" value="32" step="8">
          <label for="shap-n">SHAP 샘플 수</label>
          <input id="shap-n" type="number" min="16" max="1024" value="64" step="16">
          <label for="shap-bg">SHAP 배경 서열 수</label>
          <input id="shap-bg" type="number" min="2" max="32" value="4" step="2">
          <p id="xai-cost" class="caption"></p>
        </details>
        <fieldset>
          <legend>XAI 분석 기법 선택</legend>
          <label><input type="radio" name="xai-method" value="rollout" checked>Attention Rollout</label>
          <label><input type="radio" name="xai-method" value="ig">Integrated Gradients (IG)</label>
          <label><input type="radio" name="xai-method" value="shap">SHAP (Shapley Additive exPlanations)</label>
        </fieldset>
        <button type="submit">설명 계산</button>
      </form>
      <div data-slot="xai"><p>예측 결과가 없습니다.</p></div>
    </section>
  </main>
  <script>
    (() => {
      "use strict";
      const byId = (id) => document.getElementById(id);
      const slot = (name) => document.querySelector(`[data-slot="${name}"]`);
      const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
      let ready = false;
      let generation = 0;
      let started = false;
      let busy = false;
      let analysisDna = "";
      let operation = "prepare";
      let serial = 0;
      let activeId = null;
      let earlyEvents = [];
      let submissions = Promise.resolve();

      const updateButton = () => {
        byId("analyze").disabled = !ready || (busy && operation !== "explain");
      };
      const finish = () => {
        busy = false;
        activeId = -1;
        earlyEvents = [];
        byId("task-state").hidden = true;
        byId("task-label").textContent = "";
        byId("task-progress").removeAttribute("value");
        updateButton();
      };
      const showError = (error) => {
        byId("notice").textContent = `오류: ${error.message || error}`;
        if (!ready) byId("model-status").textContent = "모델 준비 실패";
        byId("retry-prepare").hidden = false;
        finish();
      };
      const request = (name, ...args) => {
        const token = ++serial;
        activeId = null;
        earlyEvents = [];
        operation = name;
        busy = true;
        updateButton();
        // Bridge calls can arrive on different Python threads. Preserve intent
        // order and buffer events that arrive before the request-id reply.
        submissions = submissions.then(() => window.pywebview.api[name](...args))
          .then(reply => {
            if (token !== serial) return;
            activeId = reply.request_id;
            // save_csv answers null when the user closes the save dialog: no
            // request was made, so no event will ever arrive to end this one.
            if (activeId === null) { finish(); return; }
            const events = earlyEvents;
            earlyEvents = [];
            events.forEach(event => window.xaiDesktop.onEvent(event));
          }).catch(error => { if (token === serial) showError(error); });
        return submissions;
      };

      function selectHeads() {
        const option = byId("xai-site").selectedOptions[0];
        const heads = option ? JSON.parse(option.dataset.heads) : [];
        const select = byId("xai-head");
        const previous = select.value;
        select.replaceChildren(...heads.map(head => new Option(head, head)));
        if (heads.includes(previous)) select.value = previous;
        byId("xai-head-control").hidden = heads.length <= 1;
        byId("xai-single-head").hidden = heads.length !== 1;
        byId("xai-single-head").textContent = `설명 대상: ${heads[0] || ""}`;
      }

      byId("retry-prepare").addEventListener("click", () => {
        byId("retry-prepare").hidden = true;
        byId("notice").textContent = "";
        byId("model-status").textContent = "모델 준비 중";
        request("prepare");
      });

      function explain() {
        if (!byId("xai-controls").reportValidity()) return;
        if (byId("xai-window").firstElementChild?.dataset.siteKey !== byId("xai-site").value) {
          byId("xai-window").replaceChildren();
        }
        byId("xai-cost").textContent = `현재 설정 기준 추론 횟수 — IG ${Number(byId("ig-steps").value) + 2}회, ` +
          `SHAP 약 ${Number(byId("shap-n").value) * Number(byId("shap-bg").value)}회`;
        const head = byId("xai-head").value;
        const option = byId("xai-site").selectedOptions[0];
        const scores = option ? JSON.parse(option.dataset.scores || '{}') : {};
        const score = scores[head];
        const metrics = [['설명 대상', head],
          [score?.[1] === 'percentile' ? '학습 분포 백분위' : '예측 절단 효율', score ? `${score[0].toFixed(1)}%` : '—'],
          ['PAM', option?.dataset.pam || ''], ['변이와의 거리', `${option?.dataset.distance || ''} bp`]];
        byId('xai-metrics').replaceChildren(...metrics.map(([label, value]) => {
          const div = document.createElement('div');
          const span = document.createElement('span'); span.textContent = label;
          const strong = document.createElement('strong'); strong.textContent = value;
          div.append(span, strong); return div;
        }));
        if (!head) {
          // Still supersede a previous site's calculation.
          cancelTask();
          slot("xai").textContent = "이 사이트의 PAM을 인식하는 변종이 없어 설명할 출력이 없습니다.";
          return;
        }
        const method = document.querySelector('[name="xai-method"]:checked').value;
        const params = method === "ig" ? {steps: Number(byId("ig-steps").value)} :
          method === "shap" ? {nsamples: Number(byId("shap-n").value),
            n_background: Number(byId("shap-bg").value)} : {};
        slot("xai").textContent = "설명 계산 중...";
        byId("notice").textContent = "";
        byId("task-label").textContent = method === "shap" ? "SHAP 계산 중..." :
          method === "ig" ? "IG 계산 중..." : "Attention Rollout 계산 중...";
        byId("task-progress").removeAttribute("value");
        byId("task-state").hidden = false;
        request("explain", analysisDna, byId("xai-site").value, head, method, params,
          Number(byId("graph-size").value));
      }

      function cancelTask() {
        ++serial;
        activeId = -1;
        earlyEvents = [];
        submissions = submissions.then(() => window.pywebview.api.cancel()).catch(showError);
        finish();
      }
      byId("cancel-task").addEventListener("click", () => {
        cancelTask();
        byId("notice").textContent = "계산을 취소했습니다.";
      });
      byId("xai-controls").addEventListener("submit", event => {
        event.preventDefault();
        explain();
      });
      byId("xai-controls").addEventListener("change", event => {
        if (event.target.id === "xai-site") {
          selectHeads();
          highlightDetail();
        }
        explain();
      });

      // ── 전체 상세 보기 (#185) ──────────────────────────
      // The fragment is replaced on every analysis, so its controls are reached
      // by delegation from the panel, which is part of the shell.
      let detailSearched = null;

      const detailBoxes = () =>
        Array.from(document.querySelectorAll('[name="detail-cas"]'));
      const detailRows = () =>
        Array.from(document.querySelectorAll("#detail-results tbody tr"));
      const detailFilters = () => ({
        cas_names: detailBoxes().filter(box => box.checked).map(box => box.value),
        eff_range: [Number(byId("detail-eff-low").value),
          Number(byId("detail-eff-high").value)],
        dist_range: [Number(byId("detail-dist-low").value),
          Number(byId("detail-dist-high").value)],
      });

      function highlightDetail() {
        const key = byId("xai-site").value;
        let selected = 0;
        detailRows().forEach(row => {
          const on = !row.hidden && row.dataset.siteKey === key;
          row.classList.toggle("selected", on);
          if (on) selected += 1;
        });
        const counter = byId("detail-selected");
        if (counter) counter.textContent = selected;
      }

      // Each end bounds the other so paired range sliders cannot cross.
      function detailBounds() {
        const [effLow, effHigh] = [byId("detail-eff-low"), byId("detail-eff-high")];
        const [distLow, distHigh] = [byId("detail-dist-low"), byId("detail-dist-high")];
        effLow.max = effHigh.value;
        effHigh.min = effLow.value;
        distLow.max = distHigh.value;
        distHigh.min = distLow.value;
        byId("detail-eff-caption").textContent =
          `선택 범위: ${Number(effLow.value).toFixed(1)}% ~ ${Number(effHigh.value).toFixed(1)}%`;
        byId("detail-dist-caption").textContent =
          `변이로부터 ${distLow.value} ~ ${distHigh.value} bp 이내`;
      }

      function detailSearch() {
        // Same predicate and bounds as Python's DetailFilter, on the same
        // unrounded values, so the saved CSV holds exactly these rows.
        const filters = detailFilters();
        const chosen = new Set(filters.cas_names);
        const [effLow, effHigh] = filters.eff_range;
        const [distLow, distHigh] = filters.dist_range;
        let shown = 0;
        detailRows().forEach(row => {
          const eff = Number(row.dataset.eff);
          const dist = Number(row.dataset.dist);
          // casName, not head: the filter picks a Cas variant, not a model output.
          const keep = chosen.has(row.dataset.casName) && eff >= effLow && eff <= effHigh
            && dist >= distLow && dist <= distHigh;
          row.hidden = !keep;
          if (keep) row.cells[0].textContent = ++shown;  // Streamlit renumbers too
        });
        const results = byId("detail-results");
        results.hidden = false;
        results.querySelector("table").hidden = shown === 0;
        byId("detail-highlight").hidden = shown === 0;
        byId("detail-no-match").hidden = shown > 0;
        byId("detail-count").textContent =
          `검색 결과: ${shown}개 / 전체 ${results.dataset.total}개`;
        detailSearched = shown > 0 ? filters : null;
        byId("detail-csv").disabled = shown === 0;
        highlightDetail();
      }

      function saveCsv() {
        if (!detailSearched) return;
        byId("notice").textContent = "CSV를 저장하는 중...";
        byId("task-label").textContent = "CSV 저장 준비 중...";
        byId("task-progress").removeAttribute("value");
        byId("task-state").hidden = false;
        request("save_csv", analysisDna, detailSearched);
      }

      byId("view-detail").addEventListener("submit", event => {
        event.preventDefault();
        detailSearch();
      });
      byId("view-detail").addEventListener("input", event => {
        if (event.target.type === "range") detailBounds();
      });
      byId("view-detail").addEventListener("click", event => {
        const id = event.target.id;
        if (id === "detail-all" || id === "detail-none") {
          detailBoxes().forEach(box => { box.checked = id === "detail-all"; });
        } else if (id === "detail-csv") {
          saveCsv();
        }
      });

      function selectTab(tab) {
        for (const item of tabs) {
          const selected = item === tab;
          item.setAttribute("aria-selected", String(selected));
          item.tabIndex = selected ? 0 : -1;
          byId(item.getAttribute("aria-controls")).hidden = !selected;
        }
      }
      tabs.forEach((tab, index) => {
        tab.addEventListener("click", () => selectTab(tab));
        tab.addEventListener("keydown", (event) => {
          let next;
          if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
          if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
          if (event.key === "Home") next = 0;
          if (event.key === "End") next = tabs.length - 1;
          if (next !== undefined) {
            event.preventDefault();
            selectTab(tabs[next]);
            tabs[next].focus();
          }
        });
      });

      function resizeChart() {
        const size = byId("graph-size").value;
        byId("graph-size-value").textContent = `${size}%`;
        slot("xai").querySelectorAll("figure").forEach((figure) => {
          figure.style.width = `${size}%`;
        });
      }
      byId("graph-size").addEventListener("input", resizeChart);

      byId("analysis-form").addEventListener("submit", (event) => {
        event.preventDefault();
        if (!ready || (busy && operation !== "explain")) return;
        analysisDna = byId("dna").value;
        byId("notice").textContent = "PAM 탐색 중...";
        byId("results").hidden = true;
        byId("xai-controls").hidden = true;
        byId("xai-window").replaceChildren();
        byId("task-state").hidden = true;
        detailSearched = null;  // the new analysis brings its own detail fragment
        byId('model-diagnostics')?.remove();
        for (const name of ["sites", "best", "variants", "detail", "xai"]) {
          slot(name).textContent = "";
        }
        request("scan_sites", analysisDna);
      });

      window.xaiDesktop = {
        start() {
          if (started) return;
          started = true;
          // Allow a paint first. Occluded/minimized WebViews can suspend animation
          // frames, so preparation must also have a timer fallback.
          let preparing = false;
          const prepare = () => {
            if (preparing) return;
            preparing = true;
            request("prepare");
          };
          requestAnimationFrame(() => requestAnimationFrame(prepare));
          setTimeout(prepare, 250);
        },
        onEvent(event) {
          if (event.generation !== undefined) {
            if (event.generation < generation) return;
            generation = event.generation;
          }
          if (event.type === "ready") {
            ready = true;
            byId("retry-prepare").hidden = true;
            byId("model-status").innerHTML = event.html;
            if (operation === "prepare") finish();
            else updateButton();
            return;
          }
          if (activeId === null) {
            earlyEvents.push(event);
            return;
          }
          if (event.request_id !== activeId) return;
          if (event.type === "progress") {
            if (operation === "predict" || operation === "explain" || operation === "save_csv") {
              const pct = event.total > 0 ? Math.min(100, Math.max(0, event.done / event.total * 100)) : 0;
              byId("task-progress").value = pct;
              const method = document.querySelector('[name="xai-method"]:checked').value;
              byId("task-label").textContent = operation === "predict"
                ? `모델 예측 중... ${Math.floor(pct)}%`
                : operation === "save_csv"
                ? `CSV 저장 준비 중... ${Math.floor(pct)}%`
                : `${method === "shap" ? "SHAP" : "IG"} 계산 중... ${Math.floor(pct)}%`;
            }
            return;
          }
          if (event.type === "error") {
            byId("notice").innerHTML = event.html;
            if (!ready) byId("model-status").textContent = "모델 준비 실패";
            byId("retry-prepare").hidden = false;
            finish();
            return;
          }
          if (event.type !== "result") return;
          for (const [name, html] of Object.entries(event.fragments)) {
            const target = slot(name);
            if (target) target.innerHTML = html;
          }
          const windowSummary = slot("xai").querySelector(".xai-window");
          if (windowSummary) byId("xai-window").replaceChildren(windowSummary);
          if (operation === "scan_sites") {
            byId("notice").textContent = "";
            byId("task-label").textContent = "모델 예측 중... 0%";
            byId("task-progress").value = 0;
            byId("task-state").hidden = false;
            request("predict", analysisDna);
          } else if (operation === "predict") {
            finish();
            const diagnostics = byId("model-diagnostics");
            if (diagnostics) byId("results").append(diagnostics);
            byId("welcome").hidden = true;
            byId("results").hidden = false;
            const template = byId("xai-sites-template");
            byId("xai-site").replaceChildren(...(template ? template.content.cloneNode(true).children : []));
            if (byId("xai-site").options.length) {
              byId("xai-controls").hidden = false;
              selectHeads();
              explain();
            } else {
              slot("xai").textContent = "후보 사이트가 없어 XAI를 표시할 수 없습니다.";
              byId("notice").textContent = "";
              finish();
            }
          } else {
            resizeChart();
            // save_csv reports where it saved through the notice slot itself.
            if (operation !== "save_csv") byId("notice").textContent = "";
            finish();
          }
        }
      };
    })();
  </script>
</body>
</html>
"""
