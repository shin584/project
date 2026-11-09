"""Rebuild Test CSV, method NPZs, summaries and Korean report without a model."""

import csv
import io
import json
import uuid
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from cas12a.nt_data import canonical_bytes, digest

from .nt_agreement import PROTOCOL as AGREEMENT_PROTOCOL
from .nt_agreement import sequence_agreement, summarize_population
from .nt_artifacts import prediction_metrics
from .nt_inputs import DNA_SPANS, LAYOUT
from .nt_main_artifacts import array_schema, read_main, source_roles
from .nt_run import DirectoryStore


def _csv(store, name, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    store.put(name, stream.getvalue().encode("utf-8"))


def _npz(store, name, arrays, metadata):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    checksum = store.put(name + ".npz", stream.getvalue())
    store.put(
        name + ".sidecar.json",
        canonical_bytes(
            {
                **metadata,
                "npz_sha256": checksum,
                "arrays": array_schema(arrays),
            }
        ),
    )


def _profile_svg(series, title, *, labels=None):
    width, height = 760, 300
    finite = [
        v
        for values in series.values()
        for v in values
        if v is not None and np.isfinite(v)
    ]
    maximum = max(finite, default=1) or 1
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="25" y="25" font-size="16">{escape(title)}</text>',
        '<path d="M50 50V245H735" fill="none" stroke="black"/>',
    ]
    colors = ("#0969da", "#cf222e", "#1a7f37", "#8250df")
    for offset, (name, values) in enumerate(series.items()):
        points = [
            f"{50 + 680 * i / max(1, len(values) - 1):.2f},{245 - 185 * v / maximum:.2f}"
            for i, v in enumerate(values)
            if v is not None and np.isfinite(v)
        ]
        color = colors[offset % len(colors)]
        svg.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{60 + offset * 165}" y="285" fill="{color}" font-size="12">{escape(name)}</text>'
        )
    if labels:
        for i, label in enumerate(labels):
            svg.append(
                f'<text x="{45 + 680 * i / max(1, len(labels) - 1):.1f}" y="265" font-size="11">{escape(str(label))}</text>'
            )
    svg.append(f'<text x="5" y="55" font-size="10">{maximum:.3g}</text></svg>')
    return "\n".join(svg).encode()


def _histogram_svg(values):
    counts, bounds = np.histogram(values, bins=np.linspace(-1, 1, 11))
    maximum = int(max(counts, default=0)) or 1
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="300">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="25">IG-ISM Spearman: five 6-mers, all non-WT</text>',
    ]
    for i, count in enumerate(counts):
        h = 180 * int(count) / maximum
        parts.append(
            f'<rect x="{60 + i * 65}" y="{235 - h}" width="55" height="{h}" fill="#0969da"/>'
        )
        parts.append(
            f'<text x="{60 + i * 65}" y="255" font-size="11">{bounds[i]:.1f}</text>'
        )
        parts.append(
            f'<text x="{60 + i * 65}" y="{230 - h}" font-size="11">{count}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts).encode()


def _case_explanation(method, shard):
    if shard is None:
        return {"status": "not_computed"}
    result = shard["result"]
    summary = result.get("summary")
    if method == "ism" and shard["status"] == "available":
        from .nt_ism import _summary

        summary = _summary(result["rows"][0], shard["arrays"], 0)
        sensitivity = np.asarray(summary["sensitivity"])
        total = float(sensitivity.sum())
        summary.update(
            {
                "top_position": int(np.argmax(sensitivity)),
                "token_mean_sensitivity": [
                    float(sensitivity[start:end].mean()) for start, end in DNA_SPANS
                ],
                "pam_share": float(sensitivity[4:8].sum() / total) if total else None,
                "pam_share_undefined_reason": None if total else "zero ISM sensitivity",
            }
        )
    return {
        "status": shard["status"],
        "summary": summary,
        "failure": result.get("failure"),
    }


def _group_diagnostics(rows):
    summaries = {}
    for version in ("all_nonWT", "exclude_pam_breaking"):
        summaries[version] = {}
        for group in ("length", "pure_flank"):
            available = [
                r["agreement"]["versions"][version]["groups"][group]
                for r in rows
                if r["agreement"] is not None
            ]
            fields = {}
            for key in (
                "within_contribution",
                "between_contribution",
                "contribution_sum",
                "sum_residual",
            ):
                values = [item[key] for item in available if item[key] is not None]
                fields[key] = {
                    "n_valid": len(values),
                    "mean": float(np.mean(values)) if values else None,
                }
            for axis in ("ig", "ism"):
                values = [
                    item["rank_mean_difference"][axis]
                    for item in available
                    if item["rank_mean_difference"][axis] is not None
                ]
                fields[axis + "_rank_mean_difference"] = {
                    "n_valid": len(values),
                    "mean": float(np.mean(values)) if values else None,
                }
            summaries[version][group] = {
                "n_total": len(rows),
                "n_paired": len(available),
                **fields,
            }
    return summaries


def report_main(directory, output_dir=None):
    """Write a new report snapshot from saved shards; never perform GPU inference."""
    saved = read_main(directory)
    root, plan = Path(directory).resolve(), saved["plan"]
    store = DirectoryStore(output_dir or root / "reports" / uuid.uuid4().hex[:12])
    if store.root.exists() and any(store.root.iterdir()):
        raise ValueError("report requires a new empty output directory")
    rows, shards = saved["predictions"], saved["shards"]
    cases = saved["cases"]["cases"] if saved["cases"] else []
    groups = {
        "all": rows,
        **{
            pam: [r for r in rows if r["pam_stratum"] == pam]
            for pam in ("TTTV", "TTTT")
        },
    }
    metrics = {
        group: {
            view: prediction_metrics(members, view + "_prediction")
            for view in ("raw", "clipped")
        }
        for group, members in groups.items()
    }
    flat = [
        {
            **{
                k: row[k]
                for k in (
                    "source_row",
                    "sequence",
                    "label",
                    "raw_prediction",
                    "clipped_prediction",
                    "pam_stratum",
                )
            },
            **{
                f"{view}_{key}_error": row["errors"][view][key]
                for view in ("raw", "clipped")
                for key in ("signed", "absolute", "squared")
            },
            "output_unit": "fraction",
        }
        for row in rows
    ]
    fields = [
        "source_row",
        "sequence",
        "label",
        "raw_prediction",
        "clipped_prediction",
        "pam_stratum",
        *[
            f"{v}_{e}_error"
            for v in ("raw", "clipped")
            for e in ("signed", "absolute", "squared")
        ],
        "output_unit",
    ]
    _csv(store, "predictions.csv", flat, fields)
    metadata = {
        "schema": 1,
        "partition": "test",
        "fixture": plan["fixture"],
        "input_identity": plan["input_identity"],
        "code_sha256": plan["code_sha256"],
        "execution": saved["execution"],
        "sessions": saved["sessions"],
        "plan_sha256": digest(plan),
        "layout": LAYOUT,
        "dna_spans": DNA_SPANS,
        "coordinates": "0-based half-open",
        "head_axis": None,
        "output_unit": "fraction",
        "join": "source_row; never join different methods by row offset",
        "statistics_ids": plan["statistics_ids"],
        "roles": {
            str(r["source_row"]): source_roles(plan, cases, r["source_row"])
            for r in rows
        },
    }
    store.put(
        "predictions.sidecar.json",
        canonical_bytes({**metadata, "coverage": saved["coverage"]["prediction"]}),
    )
    for method in ("ism", "ig", "rollout"):
        items = [shards[method][i] for i in sorted(shards[method])]
        array_items = (
            [s for s in items if s["status"] == "available"]
            if method == "ism"
            else items
        )
        arrays = {
            "source_row": np.asarray(
                [s["source_row"] for s in array_items], dtype=np.int64
            ),
            "accepted": np.asarray(
                [s["status"] == "available" for s in array_items], dtype=bool
            ),
        }
        if method == "ism" and array_items:
            arrays.update(
                {
                    key: np.concatenate([s["arrays"][key] for s in array_items], axis=0)
                    for key in array_items[0]["arrays"]
                    if key != "source_row"
                }
            )
        elif method == "ig":
            arrays["signed_mass"] = (
                np.stack(
                    [
                        s["arrays"]["signed_mass"][-1]
                        if s["status"] == "available"
                        else np.full(12, np.nan)
                        for s in items
                    ]
                )
                if items
                else np.empty((0, 12), dtype=np.float64)
            )
        elif method == "rollout":
            for key, shape in (("full", (10, 10)), ("sequence", (9, 9))):
                arrays[key] = (
                    np.stack(
                        [
                            s["arrays"][key]
                            if s["status"] == "available"
                            else np.full(shape, np.nan)
                            for s in items
                        ]
                    )
                    if items
                    else np.empty((0, *shape))
                )
        _npz(
            store,
            method,
            arrays,
            {
                **metadata,
                "method": method,
                "coverage": saved["coverage"][method],
                "axes": {
                    "ism": ["source_row", "position_0_to_33", "alternative_A_C_G_T"],
                    "ig": ["source_row", "CLS_9_DNA_2_padding"],
                    "rollout": ["source_row", "query_token", "key_token"],
                }[method],
                "missing": "not_computed has no row; failed ISM is in sidecar coverage only; failed IG/rollout is NaN, never zero",
                "details": [
                    {k: v for k, v in s.items() if k != "arrays"} for s in items
                ],
            },
        )
    positions, token_ig = {}, {}
    for group, members in groups.items():
        available = [
            shards["ism"][r["source_row"]]
            for r in members
            if r["source_row"] in shards["ism"]
            and shards["ism"][r["source_row"]]["status"] == "available"
        ]
        profiles = [
            np.sum(np.abs(s["arrays"]["raw_delta"][0]), axis=-1) / 3 for s in available
        ]
        positions[group] = {
            "requested": len(members),
            "n_available": len(profiles),
            "mean_abs_raw_delta": np.mean(profiles, axis=0).tolist()
            if profiles
            else [None] * 34,
        }
        selected = [r for r in members if r["source_row"] in plan["statistics_ids"]]
        ig = [
            shards["ig"][r["source_row"]]
            for r in selected
            if r["source_row"] in shards["ig"]
            and shards["ig"][r["source_row"]]["status"] == "available"
        ]
        token_ig[group] = {
            "requested": len(selected),
            "n_accepted": len(ig),
            "mean_abs_signed_mass": np.mean(
                [np.abs(s["arrays"]["signed_mass"][-1, 1:10]) for s in ig], axis=0
            ).tolist()
            if ig
            else [None] * 9,
        }
    agreement_rows = []
    predictions = {r["source_row"]: r for r in rows}
    requested = saved["coverage"]["ig"]["requested"]
    for i in requested:
        ig, ism = shards["ig"].get(i), shards["ism"].get(i)
        reasons = []
        if ig is None:
            reasons.append("IG not_computed")
        elif ig["status"] != "available":
            reasons.append(
                "IG failed: " + json.dumps(ig["result"]["failure"], ensure_ascii=False)
            )
        if ism is None:
            reasons.append("ISM not_computed")
        elif ism["status"] != "available":
            reasons.append("ISM failed: " + ism["result"]["failure"]["reason"])
        agreement = None
        if not reasons:
            agreement = sequence_agreement(
                predictions[i]["sequence"],
                ig["arrays"]["signed_mass"][-1, 1:10],
                ism["arrays"]["raw_delta"][0],
                base_valid_mask=ism["arrays"]["base_valid_mask"][0],
            )
        agreement_rows.append(
            {
                "source_row": i,
                "agreement": agreement,
                "exclusion_reasons": reasons,
                "roles": source_roles(plan, cases, i),
            }
        )
    statistics = [
        r for r in agreement_rows if r["source_row"] in plan["statistics_ids"]
    ]
    agreement_groups = {
        "all": statistics,
        **{
            pam: [
                r
                for r in statistics
                if r["source_row"] in predictions
                and predictions[r["source_row"]]["pam_stratum"] == pam
            ]
            for pam in ("TTTV", "TTTT")
        },
    }
    agreement = {
        "protocol": AGREEMENT_PROTOCOL,
        "ig_mode": plan["ig_mode"],
        "statistics_ids": plan["statistics_ids"],
        "rows": agreement_rows,
        "populations": {
            g: summarize_population(members) for g, members in agreement_groups.items()
        },
        "case_only_ids": sorted(set(requested) - set(plan["statistics_ids"])),
    }
    agreement["group_diagnostics"] = {
        g: _group_diagnostics(members) for g, members in agreement_groups.items()
    }
    annotated_cases = []
    for case in cases:
        i = case["source_row"]
        annotated_cases.append(
            {
                **case,
                "roles": source_roles(plan, cases, i),
                "methods": {
                    m: _case_explanation(m, shards[m].get(i))
                    for m in ("ism", "ig", "rollout")
                },
            }
        )
    rollout = [
        {
            "source_row": i,
            "status": s["status"],
            "summary": s["result"].get("summary"),
            "quality": s["result"].get("quality"),
            "failure": s["result"]["failure"],
        }
        for i, s in sorted(shards["rollout"].items())
    ]
    rollout_groups = {}
    for group, members in groups.items():
        accepted = [
            shards["rollout"][r["source_row"]]
            for r in members
            if r["source_row"] in shards["rollout"]
            and shards["rollout"][r["source_row"]]["status"] == "available"
        ]
        rollout_groups[group] = {
            "requested": len(members),
            "n_accepted": len(accepted),
            "mean_removed_cls_mass": np.mean(
                [s["result"]["summary"]["removed_cls_mass"] for s in accepted], axis=0
            ).tolist()
            if accepted
            else None,
            "near_identical_query_count": sum(
                s["result"]["summary"]["sequence_query_diagnostic"]["status"]
                == "near_identical_queries"
                for s in accepted
            ),
        }
    summary = {
        "schema": 1,
        "complete": saved["complete"],
        "partition": "test",
        "fixture": plan["fixture"],
        "population_count": len(plan["input_identity"]["test_source_rows"]),
        "coverage": saved["coverage"],
        "case_selection_complete": saved["case_selection_complete"],
        "ig_mode": plan["ig_mode"],
        "metrics": metrics,
        "historical_budget_status": plan["historical_budget_status"],
        "production_execution_policy": plan["production_execution_policy"],
    }
    for filename, payload in (
        ("summary.json", summary),
        ("metrics.json", metrics),
        ("position_profiles.json", positions),
        ("token_profiles.json", token_ig),
        ("agreement.json", agreement),
        (
            "cases_annotated.json",
            {"selection": saved["cases"], "cases": annotated_cases},
        ),
        ("rollout_diagnostics.json", rollout),
        ("rollout_summary.json", rollout_groups),
    ):
        store.put(filename, canonical_bytes(payload))
    _csv(
        store,
        "cases.csv",
        cases,
        [
            "source_row",
            "raw_prediction",
            "label",
            "raw_absolute_error",
            "clipping_distance",
            "pam_stratum",
            "role",
            "rank_in_category",
            "selection_reason",
        ],
    )
    store.put(
        "position_profiles.svg",
        _profile_svg(
            {g: p["mean_abs_raw_delta"] for g, p in positions.items()},
            "ISM mean |raw delta| / alternative",
            labels=None,
        ),
    )
    store.put(
        "token_profiles.svg",
        _profile_svg(
            {g: p["mean_abs_signed_mass"] for g, p in token_ig.items()},
            "IG mean absolute signed DNA token mass",
            labels=range(1, 10),
        ),
    )
    store.put(
        "agreement_distribution.svg",
        _histogram_svg(agreement["populations"]["all"]["primary"]["distribution"]),
    )
    lines = [
        "# Cas12a NT Test 본 분석 결과",
        "",
        "합성 CPU 검증 결과입니다. 실제 GPU Test 분석 증거가 아닙니다."
        if plan["fixture"]
        else "기존 내부 Testset에 대한 native FP32 단일 출력 분석입니다.",
        "",
        f"- Test 모집단: {summary['population_count']}행",
        f"- 분석 완료: {summary['complete']}",
        f"- IG 통계 모집단: {plan['ig_mode']}, {len(plan['statistics_ids'])}행. 사례 전용 ID는 통계에서 제외합니다.",
        "- raw/clipped 단위: fraction. 기존 분할·학습 완료 모델을 사용합니다.",
        "",
        "## 방법별 실제 coverage",
        "",
        "|방법|요구|성공|실패|미계산|",
        "|---|---:|---:|---:|---:|",
    ]
    for method, c in saved["coverage"].items():
        lines.append(
            f"|{method}|{len(c['requested'])}|{len(c['available'])}|{len(c['failed'])}|{len(c['not_computed'])}|"
        )
    lines += [
        "",
        "## 예측 성능",
        "",
        "|PAM 집단|view|n|Spearman|Pearson|MAE|MSE|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group, views in metrics.items():
        for view, m in views.items():
            values = [
                str(m[k]) if m[k] is not None else "undefined"
                for k in ("count", "spearman", "pearson", "mae", "mse")
            ]
            lines.append(f"|{group}|{view}|" + "|".join(values) + "|")
    primary = agreement["populations"]["all"]["primary"]
    lines += [
        "",
        "## 설명과 IG–ISM 일치도",
        "",
        f"주 통계는 같은 길이 t1–t5의 6-mer, PAM-breaking 포함, density Spearman입니다. 유효 {primary['n_valid']}/{primary['n_total']}행, 중앙값 {primary['median']}.",
        "9-token, PAM-breaking 제외, mass/span-sum, 평균 profile과 길이·순수 flank군의 global rank 및 within/between 기여는 agreement.json에 기록합니다.",
        "주 통계는 마지막 protospacer 1bp를 제외하며 PAM 혼합 토큰은 포함합니다. PAM-breaking 제외 분석은 IG와 ISM의 지지 범위가 다를 수 있습니다.",
        "낮은 상관은 설명 정확도 실패가 아닙니다. 정상 zero norm과 undefined는 이유와 함께 보존하며 실패·미계산 IG를 0으로 채우지 않습니다.",
        "IG zero reference는 분포 내 경로를 보장하지 않습니다. special attribution과 completeness/convergence는 IG sidecar에 별도로 기록합니다.",
        "",
        "![ISM 위치별 평균](position_profiles.svg)",
        "",
        "![IG 토큰별 평균](token_profiles.svg)",
        "",
        "![IG–ISM 분포](agreement_distribution.svg)",
        "",
        "## 기술 사례",
        "",
        "raw 범위 이탈→높은 오차→낮은 오차 순으로 각 최대 5개, 중복 제외·미충원입니다. 예측과 라벨로만 선정하며 설명값으로 재선정하지 않습니다.",
        "cases.csv와 cases_annotated.json에서 source ID별 선정 이유, IG/ISM/rollout의 상태와 설명을 확인할 수 있습니다.",
        "",
        "## rollout 해석",
        "",
        "encoder token attention flow입니다. CLS를 유지해 전파한 뒤 DNA 블록을 추출하며 재정규화하지 않습니다. 제거된 CLS 질량과 query 유사성 진단은 rollout_diagnostics.json에 저장합니다. 출력 attribution이나 선택적 토큰 쌍 상호작용을 입증하지 않습니다.",
        "",
        "## 범위와 한계",
        "",
        "TTTV/TTTT는 관측된 각 집단의 행 수를 분모로 사용합니다. 표본 IG의 PAM 차이는 탐색적입니다. 부분 결과는 실제 성공 coverage만 설명하며 전체 완료로 표시하지 않습니다.",
        "validation 근거와 이 Test 결과는 별도입니다. 과거 FP16 평가와 FP32의 동등성을 주장하지 않습니다.",
        "사용자 승인에 따라 본 분석은 과거 검증 실행의 시간 초과 차단과 분리했습니다. 기존 study·receipt·charge와 종료 시각은 변경하거나 정산하지 않습니다. 새 실행의 세션별 경과 시간은 별도로 기록합니다.",
        "",
    ]
    case_lines = []
    for case in annotated_cases:
        methods = case["methods"]
        status = ", ".join(
            f"{method}: {detail['status']}" for method, detail in methods.items()
        )
        description = f"source {case['source_row']}, {case['role']}: raw={case['raw_prediction']:.6g}, label={case['label']:.6g}, |error|={case['raw_absolute_error']:.6g}; {status}"
        if methods["ism"].get("summary") is not None:
            detail = methods["ism"]["summary"]
            description += f"; ISM 최대 위치={detail['top_position']} (0-based), PAM sensitivity 비중={detail['pam_share']}"
        case_lines.append("- " + description)
    index = lines.index("## rollout 해석") - 1
    lines[index:index] = ["", *case_lines]
    roll = rollout_groups["all"]
    index = lines.index("## 범위와 한계") - 1
    lines[index:index] = [
        "",
        f"rollout 품질 통과 {roll['n_accepted']}/{roll['requested']}행 중 DNA query가 거의 동일한 행은 {roll['near_identical_query_count']}행입니다.",
        "",
    ]
    store.put("report.ko.md", "\n".join(lines).encode("utf-8"))
    return {**summary, "output_dir": str(store.root)}
