"""#193의 동결된 수치 계약을 실제 두 모델로 검증한다.

`contract.json`의 기준을 **읽어서** 판정하고, 측정값은 `verification.json`에만 쓴다.
기준을 이 스크립트에서 다시 정의하지 않는다 — 측정 후에 기준이 움직일 수 없게
하는 것이 이 분리의 목적이다.

실행 (워크트리 루트에서):

    python docs/runtime/verify_contract.py init --work <new-empty-bundle> --run-id <id>
    python docs/runtime/verify_contract.py reference --work <bundle>
    python docs/runtime/verify_contract.py verify --work <bundle>
    python docs/runtime/verify_contract.py ram --work <bundle>
    python docs/runtime/verify_contract.py minimal     # 최소 실행 (스테이징/오프라인 검증용)

`reference`와 `minimal`은 별도 프로세스로 돌아야 한다. Keras backend는 프로세스마다
한 번만 정해지기 때문이다. 과거 `.packaging/issue193/` 자료는 보존한다.
새 실행과 재조립 절차는 `verification-206.md`를 따른다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from verification_bundle import (RawEvidence, initialize, load_manifest, read_json,
                                 stage_execution)
from verification_policy import all_finite, VERSIONS

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "runtime"
CONTRACT = json.loads((DOCS / "contract.json").read_text(encoding="utf-8"))
FROZEN_INPUTS = None

MODEL_KEYS = {"spcas9": "spcas9_non_nt", "cas12a": "cas12a_non_nt"}


# ════════════════════════════════════════════════
# 공통: 케이스 입력
# ════════════════════════════════════════════════

def case_windows(key: str) -> list[tuple[str, str]]:
    return [(c["id"], c["window"]) for c in CONTRACT["cases"][key]]


def case_inputs(key: str, window: str) -> list[np.ndarray]:
    """배치 1개짜리 입력 목록. torch/TF 양쪽에서 같은 바이트를 쓴다."""
    if FROZEN_INPUTS is not None:
        case_id = next(c["id"] for c in CONTRACT["cases"][key] if c["window"] == window)
        count = len(CONTRACT["models"][MODEL_KEYS[key]]["inputs"])
        return [FROZEN_INPUTS[f"{key}/{case_id}/input/{i}"].copy() for i in range(count)]
    import non_nt_runtime as rt

    seq = rt.one_hot(window)[None]
    if key == "cas12a":
        return [seq, rt.cas12a_derive(window)[None]]
    return [seq]


def weight_digest(array: np.ndarray) -> str:
    data = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(data.tobytes()).hexdigest()


# ════════════════════════════════════════════════
# 1. 기준 런타임 (TensorFlow backend)
# ════════════════════════════════════════════════

def run_reference(out_dir: Path) -> None:
    """같은 `.keras` 바이트를 TensorFlow backend로 올려 기준값을 저장한다."""
    os.environ["KERAS_BACKEND"] = "tensorflow"
    sys.path.insert(0, str(ROOT))
    import keras
    import tensorflow as tf

    assert keras.backend.backend() == "tensorflow", keras.backend.backend()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"keras": keras.__version__, "tensorflow": tf.__version__,
            "backend": keras.backend.backend(), "models": {}}
    arrays: dict[str, np.ndarray] = {}

    import model_source
    import non_nt_runtime as rt

    for key in ("spcas9", "cas12a"):
        path = model_source.resolve(rt.CHECKPOINTS[key])
        model = keras.saving.load_model(path, compile=False)
        inner = [layer for layer in model.layers
                 if not isinstance(layer, keras.layers.InputLayer)]
        probe = keras.Model(model.inputs, [layer.output for layer in inner])
        weights = {}
        for variable in model.weights:
            value = np.asarray(variable.numpy())
            weights[variable.path] = {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": weight_digest(value),
            }
        meta["models"][key] = {
            "weights": weights,
            "verified_head": CONTRACT["models"][MODEL_KEYS[key]]["verified_head"],
            "checkpoint_sha256": CONTRACT["models"][MODEL_KEYS[key]]["sha256"],
            "layer_names": [layer.name for layer in inner],
            "n_weights": len(weights),
        }
        for case_id, window in case_windows(key):
            inputs = case_inputs(key, window)
            inputs = inputs[0] if len(inputs) == 1 else inputs
            out = model(inputs, training=False)
            out = out if isinstance(out, (list, tuple)) else [out]
            arrays[f"{key}/{case_id}/outputs"] = np.concatenate(
                [np.asarray(o).reshape(-1) for o in out])
            acts = probe(inputs, training=False)
            acts = acts if isinstance(acts, (list, tuple)) else [acts]
            for layer, value in zip(inner, acts):
                arrays[f"{key}/{case_id}/layer/{layer.name}"] = np.asarray(value)

            # 교차 런타임 gradient parity용 기준값 (addendum 1).
            head = CONTRACT["models"][MODEL_KEYS[key]]["verified_head"]["index"]
            for label, factor in (("x", 1.0), ("alpha0.5", 0.5)):
                point = [np.asarray(a, dtype=np.float32) * factor
                         for a in case_inputs(key, window)]
                tensors = [tf.Variable(a) for a in point]
                with tf.GradientTape() as tape:
                    argument = tensors[0] if len(tensors) == 1 else list(tensors)
                    out = model(argument, training=False)
                    out = out if isinstance(out, (list, tuple)) else [out]
                    target = tf.reduce_sum(tf.reshape(out[head], [-1]))
                grads = tape.gradient(target, tensors)
                for position, grad in enumerate(grads):
                    arrays[f"{key}/{case_id}/grad/{label}/{position}"] = np.asarray(grad)

    np.savez_compressed(out_dir / "reference.npz", **arrays)
    (out_dir / "reference.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"reference written: {out_dir}")


# ════════════════════════════════════════════════
# 2. 판정 (torch backend)
# ════════════════════════════════════════════════

class Checks:
    """판정 결과를 모은다. 하나라도 실패하면 전체가 실패다."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def record(self, name: str, passed: bool, measured, criterion, detail: str = "") -> bool:
        self.items.append({
            "check": name, "passed": bool(passed) and all_finite(measured),
            "measured": measured, "criterion": criterion, "detail": detail,
        })
        return bool(passed)

    @property
    def failed(self) -> list[dict]:
        return [item for item in self.items if not item["passed"]]


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    """상대 차이. 기준값이 0에 가까우면 분모를 1로 둔다 (절대 기준이 따로 있다)."""
    denominator = np.maximum(np.abs(b), 1e-6)
    return float(np.max(np.abs(a - b) / denominator))


def finite_difference(runtime, inputs, head, step):
    """중앙 차분. 좌표마다 ±h를 한 배치로 묶어 전방만 쓴다."""
    flat = [np.asarray(x, dtype=np.float32).reshape(-1).copy() for x in inputs]
    sizes = [f.size for f in flat]
    rows, slots = [], []
    for which, vector in enumerate(flat):
        for index in range(vector.size):
            for sign in (+1, -1):
                perturbed = [f.copy() for f in flat]
                perturbed[which][index] += sign * step
                rows.append(perturbed)
                slots.append((which, index, sign))
    batch = [
        np.stack([row[i].reshape(np.asarray(inputs[i]).shape[1:]) for row in rows])
        for i in range(len(inputs))
    ]
    values = runtime.forward(batch, head=head, batch_size=256)
    grads = [np.zeros(size, dtype=np.float64) for size in sizes]
    for value, (which, index, sign) in zip(values, slots):
        grads[which][index] += sign * float(value) / (2 * step)
    return [g.reshape(np.asarray(x).shape[1:]) for g, x in zip(grads, inputs)]


def run_verify(work: Path) -> int:
    sys.path.insert(0, str(ROOT))
    import non_nt_runtime as rt
    from xai import explain

    reference = np.load(work / "reference.npz")
    ref_meta = json.loads((work / "reference.json").read_text(encoding="utf-8"))
    criteria = CONTRACT["criteria"]
    settings = CONTRACT["settings"]
    default = settings["user_default"]
    finer = settings["finer_verification_run"]
    checks = Checks()
    raw = RawEvidence()
    measurements: dict = {}
    started = time.time()

    versions = rt.runtime_versions()
    checks.record("runtime/backend", versions.get("backend") == "torch", versions.get("backend"), "torch")
    expected_threads = str(CONTRACT["runtime"]["threads"]["torch_num_threads"])
    checks.record("runtime/torch_threads", versions.get("torch_threads") == expected_threads,
                  versions.get("torch_threads"), expected_threads)
    for name in VERSIONS:
        expected = CONTRACT["runtime"]["versions"][name]
        checks.record(f"version/{name}", versions.get(name) == expected,
                      versions.get(name), expected)

    for key in ("spcas9", "cas12a"):
        spec = CONTRACT["models"][MODEL_KEYS[key]]
        head = spec["verified_head"]["index"]
        runtime = rt.NonNTRuntime.load(key)
        raw.observe(runtime, key)
        checks.record(f"{key}/model_identity",
                      runtime.input_names() == [i["name"] for i in spec["inputs"]]
                      and runtime.input_shapes == [tuple(i["shape"]) for i in spec["inputs"]]
                      and runtime.n_outputs == spec["n_outputs"]
                      and runtime.head_names[head] == spec["verified_head"]["name"],
                      {"inputs": runtime.input_names(), "heads": runtime.head_names}, spec)
        derive = rt.cas12a_derive if key == "cas12a" else None
        window_length = spec["inputs"][0]["shape"][0]
        per_model: dict = {"cases": {}, "verified_head": spec["verified_head"],
                           "checkpoint_sha256": spec["sha256"]}

        # ── 가중치 배치: 같은 변수 집합, 같은 모양, 비트 동일 ──
        ref_weights = ref_meta["models"][key]["weights"]
        mine = runtime.weight_digests()
        checks.record(f"{key}/weight_placement/variable_set",
                      set(mine) == set(ref_weights), len(mine),
                      criteria["weight_placement"]["requirement"],
                      detail=f"only_here={sorted(set(mine) - set(ref_weights))} "
                             f"only_reference={sorted(set(ref_weights) - set(mine))}")
        mismatched = [name for name in sorted(set(mine) & set(ref_weights))
                      if mine[name] != ref_weights[name]]
        checks.record(f"{key}/weight_placement/bitwise", not mismatched,
                      {"n_variables": len(mine), "mismatched": mismatched},
                      criteria["weight_placement"]["requirement"])
        per_model["n_weights"] = len(mine)

        # ── 케이스별 전방 parity, 층별 parity, 함수 일치, gradient ──
        forward_abs, forward_rel, layer_worst = [], [], []
        for case_id, window in case_windows(key):
            inputs = case_inputs(key, window)
            case: dict = {"window": window}

            all_heads = np.array([
                float(runtime.forward(inputs, head=h, batch_size=1)[0])
                for h in range(runtime.n_outputs)
            ])
            ref_out = reference[f"{key}/{case_id}/outputs"]
            raw.add(f"{key}/{case_id}/outputs", all_heads)
            checks.record(f"{key}/{case_id}/output_set",
                          all_heads.shape == ref_out.shape == (spec["n_outputs"],),
                          list(all_heads.shape), spec["n_outputs"])
            checks.record(f"{key}/{case_id}/finiteness/forward",
                          all_finite([all_heads, ref_out]), True, "all finite")
            forward_abs.append(float(np.max(np.abs(all_heads - ref_out))))
            forward_rel.append(_rel(all_heads, ref_out))
            case["raw_outputs"] = [float(v) for v in all_heads]
            case["reference_outputs"] = [float(v) for v in ref_out]
            case["forward_max_abs_diff"] = forward_abs[-1]
            case["forward_max_rel_diff"] = forward_rel[-1]

            worst = {"layer": None, "abs": 0.0, "rel": 0.0, "scaled_bound": 0.0}
            layer_rows = {}
            activations = runtime.layer_activations(inputs)
            expected_layers = ref_meta["models"][key]["layer_names"]
            checks.record(f"{key}/{case_id}/layerwise/layer_set",
                          set(activations) == set(expected_layers),
                          sorted(activations), sorted(expected_layers))
            for layer_name, got in activations.items():
                raw.add(f"{key}/{case_id}/layer/{layer_name}", got)
                if layer_name not in expected_layers:
                    continue
                want = reference[f"{key}/{case_id}/layer/{layer_name}"]
                abs_diff = float(np.max(np.abs(got - want)))
                scale = float(max(1.0, np.max(np.abs(want))))
                layer_rows[layer_name] = {
                    "max_abs_diff": abs_diff, "max_rel_diff": _rel(got, want),
                    "scaled_bound": 1e-4 * scale,
                }
                checks.record(f"{key}/{case_id}/layerwise/{layer_name}/abs_scaled",
                              got.shape == want.shape and all_finite([got, want])
                              and abs_diff <= 1e-4 * scale,
                              abs_diff, 1e-4 * scale)
                checks.record(f"{key}/{case_id}/layerwise/{layer_name}/rel",
                              got.shape == want.shape and all_finite([got, want])
                              and _rel(got, want) <= criteria["layerwise_parity"]["max_rel_diff"],
                              _rel(got, want), criteria["layerwise_parity"]["max_rel_diff"])
                if abs_diff > worst["abs"]:
                    worst = {"layer": layer_name, "abs": abs_diff,
                             "rel": layer_rows[layer_name]["max_rel_diff"],
                             "scaled_bound": 1e-4 * scale}
            layer_worst.append(worst)
            case["layerwise_worst"] = worst
            case["layerwise"] = layer_rows

            # 함수 일치: IG 경로의 전방값 vs 예측 전방값
            y_grad, grads = runtime.gradients(inputs, head=head)
            y_forward = runtime.forward(inputs, head=head, batch_size=1)
            function_gap = abs(float(y_grad[0]) - float(y_forward[0]))
            checks.record(f"{key}/{case_id}/function_match",
                          function_gap <= criteria["function_match"]["max_abs_diff"],
                          function_gap, criteria["function_match"]["max_abs_diff"])
            case["function_match_abs_diff"] = function_gap
            case["all_finite_gradients"] = bool(all(np.isfinite(g).all() for g in grads))

            # gradient: 실제 원핫 점과 내부 보간점(alpha=0.5)
            fd_rows = {}
            for label, point in (("x", inputs), ("alpha0.5", [0.5 * x for x in inputs])):
                autograd = runtime.gradients(point, head=head)[1]
                numeric = finite_difference(runtime, point, head,
                                            criteria["gradient_check"]["fd_step"])
                for index, (analytic, difference) in enumerate(zip(autograd, numeric)):
                    raw.add(f"{key}/{case_id}/grad/{label}/{index}", analytic)
                    raw.add(f"{key}/{case_id}/fd/{label}/{index}", difference)
                # 기준은 "every model input"이다. 두 입력을 이어 붙여 하나로 재면
                # 화면이 염기별로 그리는 서열 gradient가 크기가 열 배인 특징
                # gradient 뒤에 숨는다. 입력별로 따로 판정하고, 합친 값도 함께 남긴다.
                names = runtime.input_names()
                per_input = {}
                for name, mine, theirs in zip(names, autograd, numeric):
                    a = np.asarray(mine, dtype=np.float64).reshape(-1)
                    b = np.asarray(theirs, dtype=np.float64).reshape(-1)
                    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                    rel_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
                    per_input[name] = {"cosine": cosine, "rel_l2_error": rel_l2,
                                       "autograd_norm": float(np.linalg.norm(a)),
                                       "fd_norm": float(np.linalg.norm(b))}
                    checks.record(
                        f"{key}/{case_id}/gradient_check/{label}/{name}/cosine",
                        cosine >= criteria["gradient_check"]["min_cosine"],
                        cosine, criteria["gradient_check"]["min_cosine"])
                    checks.record(
                        f"{key}/{case_id}/gradient_check/{label}/{name}/rel_l2",
                        rel_l2 <= criteria["gradient_check"]["max_rel_l2_error"],
                        rel_l2, criteria["gradient_check"]["max_rel_l2_error"])
                a = np.concatenate([g.reshape(-1) for g in autograd]).astype(np.float64)
                b = np.concatenate([g.reshape(-1) for g in numeric]).astype(np.float64)
                fd_rows[label] = {
                    "per_input": per_input,
                    "pooled_cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                    "pooled_rel_l2_error": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                }
            case["gradient_check"] = fd_rows

            # ── IG: 사용자 기본 32단계와 정밀 256단계 ──
            handle = rt.explain_handle(runtime, head=head, window_length=window_length,
                                       derive=derive)
            ig_rows = {}
            attributions = {}
            for label, steps in (("default", default["ig_steps"]), ("finer", finer["ig_steps"])):
                project = handle.project

                def retain_attributions(attrs, project=project):
                    for index, value in enumerate(attrs):
                        raw.add(f"{key}/{case_id}/ig/{label}/input_attribution/{index}", value)
                    return project(attrs)

                handle.project = retain_attributions
                result = explain.integrated_gradients(handle, window, steps=steps)
                handle.project = project
                raw.add(f"{key}/{case_id}/ig/{label}/attribution", result.attribution)
                terms = [result.pred, result.baseline_pred, result.completeness_error]
                raw.add(f"{key}/{case_id}/ig/{label}/terms", terms)
                totals = list((handle.input_totals or {}).values())
                raw.add(f"{key}/{case_id}/ig/{label}/input_totals", totals)
                attributions[label] = result.attribution
                bound = criteria["ig_completeness"]["max_abs_error"][key]
                ig_rows[label] = {
                    "steps": steps, "pred": result.pred,
                    "baseline_pred": result.baseline_pred,
                    "completeness_error": result.completeness_error,
                    "completeness_bound": bound,
                    "input_totals": dict(handle.input_totals or {}),
                    "all_finite": bool(np.isfinite(result.attribution).all()),
                }
                checks.record(f"{key}/{case_id}/ig_completeness/{label}",
                              result.completeness_error <= bound,
                              result.completeness_error, bound)
                checks.record(f"{key}/{case_id}/finiteness/ig_{label}",
                              all_finite([result.attribution, terms, totals]),
                              ig_rows[label]["all_finite"], "all finite")

            coarse, fine = attributions["default"], attributions["finer"]
            scale = float(np.max(np.abs(fine)))
            pearson = float(np.corrcoef(coarse, fine)[0, 1])
            max_diff = float(np.max(np.abs(coarse - fine)))
            strong = np.abs(fine) >= 0.1 * scale
            sign_agreement = float(np.mean(np.sign(coarse[strong]) == np.sign(fine[strong]))) \
                if strong.any() else 1.0
            stability = criteria["ig_resolution_stability"]
            checks.record(f"{key}/{case_id}/ig_resolution/pearson",
                          pearson >= stability["min_pearson_r"], pearson,
                          stability["min_pearson_r"])
            checks.record(f"{key}/{case_id}/ig_resolution/max_abs_diff",
                          max_diff <= 0.05 * scale, {"max_abs_diff": max_diff,
                          "bound": 0.05 * scale}, stability["max_abs_diff_scaled"])
            checks.record(f"{key}/{case_id}/ig_resolution/sign_agreement",
                          sign_agreement == 1.0, sign_agreement,
                          stability["sign_agreement"])
            ig_rows["resolution"] = {"pearson_r": pearson, "max_abs_diff": max_diff,
                                     "max_abs_diff_bound": 0.05 * scale,
                                     "sign_agreement": sign_agreement,
                                     "n_strong_positions": int(strong.sum())}
            case["ig"] = ig_rows

            # ── SHAP: 기본 설정, 같은 seed 재현, seed 교차, 정밀 실행 ──
            shap_rows = {}
            runs = {}
            for seed in (0, 1, 2):
                runs[seed] = explain.kernel_shap(
                    handle, window, nsamples=default["shap_nsamples"],
                    n_background=default["shap_n_background"], seed=seed)
            repeat = explain.kernel_shap(
                handle, window, nsamples=default["shap_nsamples"],
                n_background=default["shap_n_background"], seed=0)
            heavy = explain.kernel_shap(
                handle, window, nsamples=finer["shap_nsamples"],
                n_background=finer["shap_n_background"], seed=0)
            for run_name, result in [(f"seed{s}", runs[s]) for s in (0, 1, 2)] + [
                    ("repeat", repeat), ("finer", heavy)]:
                terms = [result.base_value, result.pred, result.additivity_error]
                raw.add(f"{key}/{case_id}/shap/{run_name}/values", result.values)
                raw.add(f"{key}/{case_id}/shap/{run_name}/terms", terms)
                checks.record(f"{key}/{case_id}/finiteness/shap_{run_name}",
                              all_finite([result.values, terms]), True, "all finite")

            additivity_bound = criteria["shap_additivity"]["max_abs_error"][key]
            checks.record(f"{key}/{case_id}/shap_additivity",
                          runs[0].additivity_error <= additivity_bound,
                          runs[0].additivity_error, additivity_bound)
            checks.record(f"{key}/{case_id}/finiteness/shap",
                          bool(np.isfinite(runs[0].values).all()
                               and np.isfinite([runs[0].base_value, runs[0].pred]).all()),
                          True, "all finite")

            determinism = float(np.max(np.abs(runs[0].values - repeat.values)))
            checks.record(f"{key}/{case_id}/shap_same_seed_determinism",
                          determinism <= criteria["shap_repeat_stability"]
                          ["same_seed_determinism"]["max_abs_diff"], determinism,
                          criteria["shap_repeat_stability"]["same_seed_determinism"]["max_abs_diff"])

            cross = criteria["shap_repeat_stability"]["cross_seed"]
            pairs = [(0, 1), (0, 2), (1, 2)]
            correlations = {f"{a}-{b}": float(np.corrcoef(runs[a].values, runs[b].values)[0, 1])
                            for a, b in pairs}
            cross_diff = max(float(np.max(np.abs(runs[a].values - runs[b].values)))
                             for a, b in pairs)
            cross_bound = 0.25 * float(np.max(np.abs(runs[0].values)))
            checks.record(f"{key}/{case_id}/shap_cross_seed/pearson",
                          min(correlations.values()) >= cross["min_pairwise_pearson_r"],
                          correlations, cross["min_pairwise_pearson_r"])
            checks.record(f"{key}/{case_id}/shap_cross_seed/max_abs_diff",
                          cross_diff <= cross_bound,
                          {"max_abs_diff": cross_diff, "bound": cross_bound},
                          cross["max_abs_diff_scaled"])

            heavy_r = float(np.corrcoef(runs[0].values, heavy.values)[0, 1])
            checks.record(f"{key}/{case_id}/shap_default_vs_finer",
                          heavy_r >= criteria["shap_repeat_stability"]["default_vs_finer"]["min_pearson_r"],
                          heavy_r,
                          criteria["shap_repeat_stability"]["default_vs_finer"]["min_pearson_r"])

            shap_rows = {
                "default": {"base_value": runs[0].base_value, "pred": runs[0].pred,
                            "additivity_error": runs[0].additivity_error,
                            "additivity_bound": additivity_bound,
                            "max_abs_value": float(np.max(np.abs(runs[0].values)))},
                "same_seed_max_abs_diff": determinism,
                "cross_seed_pearson": correlations,
                "cross_seed_max_abs_diff": cross_diff,
                "cross_seed_bound": cross_bound,
                "default_vs_finer_pearson": heavy_r,
                "finer": {"nsamples": finer["shap_nsamples"],
                          "n_background": finer["shap_n_background"],
                          "additivity_error": heavy.additivity_error},
            }
            case["shap"] = shap_rows
            per_model["cases"][case_id] = case

        checks.record(f"{key}/forward_parity/max_abs_diff",
                      max(forward_abs) <= criteria["forward_parity"]["max_abs_diff"],
                      max(forward_abs), criteria["forward_parity"]["max_abs_diff"])
        checks.record(f"{key}/forward_parity/max_rel_diff",
                      max(forward_rel) <= criteria["forward_parity"]["max_rel_diff"],
                      max(forward_rel), criteria["forward_parity"]["max_rel_diff"])
        worst_layer = max(layer_worst, key=lambda w: w["abs"])
        checks.record(f"{key}/layerwise_parity/abs_scaled",
                      all(row["max_abs_diff"] <= row["scaled_bound"]
                          for c in per_model["cases"].values() for row in c["layerwise"].values()),
                      worst_layer, criteria["layerwise_parity"]["max_abs_diff_scaled"])
        checks.record(f"{key}/layerwise_parity/rel",
                      all(row["max_rel_diff"] <= criteria["layerwise_parity"]["max_rel_diff"]
                          for c in per_model["cases"].values() for row in c["layerwise"].values()),
                      max(w["rel"] for w in layer_worst),
                      criteria["layerwise_parity"]["max_rel_diff"])
        per_model["ops_present"] = runtime.ops_present()
        per_model["layer_names"] = runtime.layer_names()
        per_model["input_names"] = runtime.input_names()
        per_model["head_names"] = runtime.head_names
        measurements[key] = per_model

    checks.record("finiteness/raw", all_finite(raw.arrays), True, "all finite")
    raw.save(work / "core-raw.npz")
    report = {
        "schema_version": 2,
        "issue": 206,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "contract_sha256": hashlib.sha256(
            (work / "snapshot/contract.json").read_bytes()).hexdigest(),
        "runtime": rt.runtime_versions(),
        "reference_runtime": {k: ref_meta[k] for k in ("keras", "tensorflow", "backend")},
        "tensorflow_importable_in_this_process": rt.tensorflow_present(),
        "elapsed_seconds": round(time.time() - started, 1),
        "status": "pass" if not checks.failed else "fail",
        "n_checks": len(checks.items),
        "n_failed": len(checks.failed),
        "failed": checks.failed,
        "checks": checks.items,
        "measurements": measurements,
    }
    (work / "verification-core.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_checks": report["n_checks"],
                      "n_failed": report["n_failed"],
                      "failed": [f["check"] for f in checks.failed]},
                     indent=2))
    return 0 if not checks.failed else 1


# ════════════════════════════════════════════════
# 3. RSS (모델별 하위 프로세스)
# ════════════════════════════════════════════════

RAM_CHILD = r'''
import json, os, sys
sys.path.insert(0, %(root)r)
import psutil
process = psutil.Process()
def rss():
    return int(process.memory_info().rss)
stages = {"interpreter": rss()}
import non_nt_runtime as rt
rt.import_keras()
stages["runtime_import"] = rss()
runtime = rt.NonNTRuntime.load(%(key)r)
stages["model_loaded"] = rss()
window = %(window)r
head = %(head)d
inputs = [rt.one_hot(window)[None]]
if %(key)r == "cas12a":
    inputs.append(rt.cas12a_derive(window)[None])
y = runtime.forward(inputs, head=head)
stages["after_forward"] = rss()
from xai import explain
handle = rt.explain_handle(runtime, head=head, window_length=len(window),
                           derive=rt.cas12a_derive if %(key)r == "cas12a" else None)
ig = explain.integrated_gradients(handle, window, steps=%(steps)d)
import numpy as np
stages["all_finite"] = bool(np.isfinite(y).all() and np.isfinite(ig.attribution).all()
    and np.isfinite([ig.pred, ig.baseline_pred, ig.completeness_error]).all())
stages["held_models"] = [%(key)r]
stages["after_ig"] = rss()
stages["peak_working_set"] = int(process.memory_info().peak_wset)
stages["tensorflow_importable"] = rt.tensorflow_present()
stages["tensorflow_imported"] = "tensorflow" in sys.modules
print("RSSJSON" + json.dumps(stages))
'''


def run_ram(work: Path) -> int:
    """프로세스 하나에 한 모델만 올려 import/적재/IG 단계별 RSS를 재는다."""
    out = {}
    for key in ("spcas9", "cas12a"):
        spec = CONTRACT["models"][MODEL_KEYS[key]]
        window = CONTRACT["cases"][key][0]["window"]
        code = RAM_CHILD % {"root": str(ROOT), "key": key, "window": window,
                            "head": spec["verified_head"]["index"],
                            "steps": CONTRACT["settings"]["user_default"]["ig_steps"]}
        started = time.time()
        finished = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                  text=True, cwd=str(ROOT))
        line = [l for l in finished.stdout.splitlines() if l.startswith("RSSJSON")]
        if not line:
            stages = {"status": "fail", "stderr": finished.stderr[-2000:]}
        else:
            stages = json.loads(line[0][len("RSSJSON"):])
        stages["command"] = finished.args
        stages["exit_code"] = finished.returncode
        stages["elapsed_seconds"] = round(time.time() - started, 1)
        out[key] = stages
    work.mkdir(parents=True, exist_ok=True)
    (work / "ram.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if all(row.get("all_finite") and row.get("exit_code") == 0 for row in out.values()) else 1


# ════════════════════════════════════════════════
# 4. 최소 실행 (스테이징·오프라인 검증에서 호출)
# ════════════════════════════════════════════════

def run_minimal() -> int:
    """두 모델을 올려 전방 1회와 IG gradient 1회만 수행한다.

    오프라인/TensorFlow 없는 스테이징에서 이 경로가 도는지만 본다.
    판정 기준은 유한값과 head 수뿐이다 — parity 판정은 `verify`의 몫이다.
    """
    sys.path.insert(0, str(ROOT))
    import non_nt_runtime as rt

    out = {"tensorflow_importable": rt.tensorflow_present(),
           "versions": rt.runtime_versions(), "models": {}}
    for key in ("spcas9", "cas12a"):
        spec = CONTRACT["models"][MODEL_KEYS[key]]
        head = spec["verified_head"]["index"]
        window = CONTRACT["cases"][key][0]["window"]
        runtime = rt.NonNTRuntime.load(key)
        inputs = case_inputs(key, window)
        y = runtime.forward(inputs, head=head)
        _, grads = runtime.gradients(inputs, head=head)
        out["models"][key] = {
            "window": window,
            "raw_output": float(y[0]),
            "n_outputs": runtime.n_outputs,
            "input_shapes": [list(s) for s in runtime.input_shapes],
            "gradient_abs_sum": [float(np.abs(g).sum()) for g in grads],
            "all_finite": bool(np.isfinite(y).all()
                               and all(np.isfinite(g).all() for g in grads)),
        }
    out["tensorflow_imported"] = "tensorflow" in sys.modules
    ok = all(m["all_finite"] and all(s > 0 for s in m["gradient_abs_sum"])
             for m in out["models"].values())
    out["status"] = "pass" if ok else "fail"
    print("MINIMALJSON" + json.dumps(out))
    return 0 if ok else 1


# ════════════════════════════════════════════════
# 5. gradient 진단 (addendum 1)
# ════════════════════════════════════════════════

def run_diagnose(work: Path) -> int:
    """실패한 좌표별 차분 검사의 원인을 가린다. 기준은 contract.json의 addendum 1.

    (1) 같은 그래프의 두 독립 autograd 구현(torch vs TensorFlow)을 맞대 보고,
    (2) 꺾임 하나의 비중이 작은 무작위 방향으로 방향 미분을 차분과 비교하며,
    (3) 좌표 섭동이 ReLU/LeakyReLU의 부호를 몇 번 넘는지 센다.
    """
    sys.path.insert(0, str(ROOT))
    import non_nt_runtime as rt

    reference = np.load(work / "reference.npz")
    addendum = CONTRACT["addendum_1"]
    parity_criteria = addendum["cross_runtime_gradient_parity"]
    directional_criteria = addendum["directional_finite_difference"]
    checks = Checks()
    raw = RawEvidence()
    out: dict = {}

    for key in ("spcas9", "cas12a"):
        spec = CONTRACT["models"][MODEL_KEYS[key]]
        head = spec["verified_head"]["index"]
        runtime = rt.NonNTRuntime.load(key)
        raw.observe(runtime, key)
        rows: dict = {}
        kink_layers = runtime.nonsmooth_layer_names()

        for case_id, window in case_windows(key):
            case: dict = {}
            for label, factor in (("x", 1.0), ("alpha0.5", 0.5)):
                point = [np.asarray(a, dtype=np.float32) * factor
                         for a in case_inputs(key, window)]
                mine = runtime.gradients(point, head=head)[1]
                for index, value in enumerate(mine):
                    raw.add(f"{key}/{case_id}/grad/{label}/{index}", value)
                theirs = [reference[f"{key}/{case_id}/grad/{label}/{i}"]
                          for i in range(len(mine))]
                # `gradient_check`과 같은 이유로 입력별로 판정한다.
                per_input = {}
                for name, torch_grad, tf_grad in zip(runtime.input_names(), mine, theirs):
                    a = np.asarray(torch_grad, dtype=np.float64).reshape(-1)
                    b = np.asarray(tf_grad, dtype=np.float64).reshape(-1)
                    max_abs = float(np.max(np.abs(a - b)))
                    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                    per_input[name] = {
                        "max_abs_diff": max_abs, "cosine": cosine,
                        "torch_norm": float(np.linalg.norm(a)),
                        "tensorflow_norm": float(np.linalg.norm(b)),
                    }
                    checks.record(
                        f"{key}/{case_id}/cross_runtime_gradient/{label}/{name}/max_abs_diff",
                        max_abs <= parity_criteria["max_abs_diff"], max_abs,
                        parity_criteria["max_abs_diff"])
                    checks.record(
                        f"{key}/{case_id}/cross_runtime_gradient/{label}/{name}/cosine",
                        cosine >= parity_criteria["min_cosine"], cosine,
                        parity_criteria["min_cosine"])
                case[f"cross_runtime_{label}"] = per_input

            # 방향 미분 vs 방향 차분 (내부 보간점)
            point = [np.asarray(a, dtype=np.float32) * 0.5
                     for a in case_inputs(key, window)]
            grads = runtime.gradients(point, head=head)[1]
            rng = np.random.default_rng(193)
            step = 1e-3
            errors = []
            directional_terms = []
            for _ in range(32):
                directions = [rng.standard_normal(np.asarray(p).shape).astype(np.float32)
                              for p in point]
                norm = np.sqrt(sum(float((d ** 2).sum()) for d in directions))
                directions = [d / norm for d in directions]
                analytic = sum(float((g * d).sum()) for g, d in zip(grads, directions))
                plus = runtime.forward([p + step * d for p, d in zip(point, directions)],
                                       head=head, batch_size=1)[0]
                minus = runtime.forward([p - step * d for p, d in zip(point, directions)],
                                        head=head, batch_size=1)[0]
                numeric = (float(plus) - float(minus)) / (2 * step)
                errors.append(abs(analytic - numeric) / max(abs(numeric), 1e-6))
                directional_terms.append([analytic, numeric, errors[-1]])
            raw.add(f"{key}/{case_id}/directional/terms", directional_terms)
            median = float(np.median(errors))
            worst = float(np.max(errors))
            checks.record(f"{key}/{case_id}/directional_fd/median",
                          median <= directional_criteria["max_median_rel_error"], median,
                          directional_criteria["max_median_rel_error"])
            checks.record(f"{key}/{case_id}/directional_fd/max",
                          worst <= directional_criteria["max_rel_error"], worst,
                          directional_criteria["max_rel_error"])
            case["directional_fd"] = {"median_rel_error": median, "max_rel_error": worst,
                                      "n_directions": len(errors), "step": step}

            # 꺾임 증거: 좌표 ±h 섭동이 ReLU/LeakyReLU 부호를 넘는 횟수
            if kink_layers:
                flips = 0
                total = 0
                base_inputs = case_inputs(key, window)
                flat_sizes = [np.asarray(x).size for x in base_inputs]
                h = CONTRACT["criteria"]["gradient_check"]["fd_step"]

                def kink_signs(arrays, runtime=runtime, kink_layers=kink_layers):
                    acts = runtime.layer_activations(arrays)
                    for name in kink_layers:
                        raw.add(f"{key}/{case_id}/kink/{len(raw.arrays)}/{name}", acts[name])
                    return np.concatenate([acts[name].reshape(-1) > 0
                                           for name in kink_layers])

                for which, size in enumerate(flat_sizes):
                    for index in range(size):
                        signs = []
                        for sign in (+1, -1):
                            perturbed = [np.array(x, dtype=np.float32, copy=True)
                                         for x in base_inputs]
                            perturbed[which].reshape(-1)[index] += sign * h
                            signs.append(kink_signs(perturbed))
                        flips += int(np.sum(signs[0] != signs[1]))
                        total += signs[0].size
                case["kink_sign_flips"] = {"flips": flips, "activation_units_probed": total,
                                           "fd_step": h,
                                           "layers_probed": list(kink_layers)}
            rows[case_id] = case
        out[key] = rows

    checks.record("finiteness/raw", all_finite(raw.arrays), True, "all finite")
    raw.save(work / "diagnose-raw.npz")
    report = {
        "schema_version": 2, "issue": 206,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "registered_criteria": "contract.json addendum_1",
        "status": "pass" if not checks.failed else "fail",
        "n_checks": len(checks.items), "n_failed": len(checks.failed),
        "failed": checks.failed, "checks": checks.items, "measurements": out,
    }
    (work / "diagnose.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_checks": report["n_checks"],
                      "failed": [f["check"] for f in checks.failed]}, indent=2))
    return 0 if not checks.failed else 1


def run_investigation(work: Path) -> int:
    """Execute a preregistered producer, or explicitly retain unverified work.

    Candidate/float64 measurement producers are implemented in #210/#211.
    They write investigations.json and investigation-raw.npz in this bundle.
    A command must be registered in the additive protocol before init; its
    script(s) must be among --extra-code or the snapshotted runtime code.
    """
    protocol = read_json(work / "snapshot/protocol.json")
    command = protocol.get("investigation_command")
    if command:
        argv = [part.replace("{python}", sys.executable).replace("{work}", str(work))
                for part in command]
        manifest = load_manifest(work, live=True)
        frozen_sources = {v["source"] for v in manifest["sources"].values()}
        if (len(argv) < 2 or Path(argv[0]).resolve() != Path(sys.executable).resolve()
                or str(Path(argv[1]).resolve()) not in frozen_sources
                or Path(argv[1]).suffix != ".py"):
            raise ValueError("Investigation producer must be frozen before measurement")
        finished = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        (work / "investigation-stdout.txt").write_text(finished.stdout, encoding="utf-8")
        (work / "investigation-stderr.txt").write_text(finished.stderr, encoding="utf-8")
        path = work / "investigations.json"
        if path.exists():
            report = read_json(path)
            report["execution"] = {"command": argv, "exit_code": finished.returncode,
                                   "executable": sys.executable}
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return finished.returncode
    report = {"investigations": [
        {"id": name, "state": "unverified", "planned": definition["planned"],
         "completed": 0, "raw_arrays": [], "reason": "Measurement producer not yet registered"}
        for name, definition in protocol["investigations"].items()]}
    (work / "investigations.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(work / "investigation-raw.npz")
    # Stage collection finished; Q17 still reports incomplete/nonzero in assembly.
    return 0


def main() -> int:
    global CONTRACT, FROZEN_INPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("init", "reference", "verify", "ram", "minimal", "diagnose", "investigation"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--contract", type=Path, default=DOCS / "contract.json")
    parser.add_argument("--protocol", type=Path, default=DOCS / "protocol-206.json")
    parser.add_argument("--extra-code", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.step == "minimal":
        return run_minimal()
    if args.work is None:
        parser.error("--work is required; historical default output is preserved")
    work = args.work.resolve()
    try:
        if args.step == "init":
            if not args.run_id:
                parser.error("init requires --run-id")
            manifest = initialize(work, args.run_id, args.contract, args.protocol, args.extra_code)
            print(json.dumps({"run_id": manifest["run_id"], "work": str(work)}))
            return 0
        load_manifest(work)
        CONTRACT = read_json(work / "snapshot/contract.json")
        with np.load(work / "snapshot/inputs.npz", allow_pickle=False) as frozen:
            FROZEN_INPUTS = {name: frozen[name] for name in frozen.files}
        with stage_execution(work, args.step) as result:
            if args.step == "reference":
                run_reference(work)
                code = 0
            elif args.step == "verify":
                code = run_verify(work)
            elif args.step == "diagnose":
                code = run_diagnose(work)
            elif args.step == "investigation":
                code = run_investigation(work)
            else:
                code = run_ram(work)
            result["exit_code"] = code
        return code
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
