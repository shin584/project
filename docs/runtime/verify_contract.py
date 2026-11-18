"""#193의 동결된 수치 계약을 실제 두 모델로 검증한다.

`contract.json`의 기준을 **읽어서** 판정하고, 측정값은 `verification.json`에만 쓴다.
기준을 이 스크립트에서 다시 정의하지 않는다 — 측정 후에 기준이 움직일 수 없게
하는 것이 이 분리의 목적이다.

실행 (워크트리 루트에서):

    python docs/runtime/verify_contract.py reference   # TensorFlow backend 기준값 생성
    python docs/runtime/verify_contract.py verify      # torch backend로 판정, 결과 기록
    python docs/runtime/verify_contract.py ram         # 프로세스 RSS 측정 (모델별 하위 프로세스)
    python docs/runtime/verify_contract.py minimal     # 최소 실행 (스테이징/오프라인 검증용)

`reference`와 `minimal`은 별도 프로세스로 돌아야 한다. Keras backend는 프로세스마다
한 번만 정해지기 때문이다. 중간 산출물은 Git 제외 경로 `.packaging/issue193/`에 둔다.
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

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "runtime"
WORK = ROOT / ".packaging" / "issue193"
CONTRACT = json.loads((DOCS / "contract.json").read_text(encoding="utf-8"))

MODEL_KEYS = {"spcas9": "spcas9_non_nt", "cas12a": "cas12a_non_nt"}


# ════════════════════════════════════════════════
# 공통: 케이스 입력
# ════════════════════════════════════════════════

def case_windows(key: str) -> list[tuple[str, str]]:
    return [(c["id"], c["window"]) for c in CONTRACT["cases"][key]]


def case_inputs(key: str, window: str) -> list[np.ndarray]:
    """배치 1개짜리 입력 목록. torch/TF 양쪽에서 같은 바이트를 쓴다."""
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
            "check": name, "passed": bool(passed),
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
    measurements: dict = {}
    started = time.time()

    versions = rt.runtime_versions()
    for name, expected in CONTRACT["runtime"]["versions"].items():
        if name in versions:
            checks.record(f"version/{name}", versions[name].startswith(expected.split("+")[0]),
                          versions[name], expected)

    for key in ("spcas9", "cas12a"):
        spec = CONTRACT["models"][MODEL_KEYS[key]]
        head = spec["verified_head"]["index"]
        runtime = rt.NonNTRuntime.load(key)
        derive = rt.cas12a_derive if key == "cas12a" else None
        window_length = spec["inputs"][0]["shape"][0]
        per_model: dict = {"cases": {}}

        # ── 가중치 배치: 같은 변수 집합, 같은 모양, 비트 동일 ──
        ref_weights = ref_meta["models"][key]["weights"]
        mine = {}
        for variable in runtime._model.weights:
            value = np.asarray(variable.numpy())
            mine[variable.path] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                   "sha256": weight_digest(value)}
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
            forward_abs.append(float(np.max(np.abs(all_heads - ref_out))))
            forward_rel.append(_rel(all_heads, ref_out))
            case["raw_outputs"] = [float(v) for v in all_heads]
            case["reference_outputs"] = [float(v) for v in ref_out]
            case["forward_max_abs_diff"] = forward_abs[-1]
            case["forward_max_rel_diff"] = forward_rel[-1]

            keras = rt.import_keras()
            inner = [layer for layer in runtime._model.layers
                     if not isinstance(layer, keras.layers.InputLayer)]
            probe = keras.Model(runtime._model.inputs,
                                [layer.output for layer in inner])
            argument = inputs[0] if len(inputs) == 1 else inputs
            acts = probe(argument, training=False)
            acts = acts if isinstance(acts, (list, tuple)) else [acts]
            worst = {"layer": None, "abs": 0.0, "rel": 0.0, "scaled_bound": 0.0}
            layer_rows = {}
            for layer, value in zip(inner, acts):
                got = np.asarray(value.detach().numpy() if hasattr(value, "detach") else value)
                want = reference[f"{key}/{case_id}/layer/{layer.name}"]
                abs_diff = float(np.max(np.abs(got - want)))
                scale = float(max(1.0, np.max(np.abs(want))))
                layer_rows[layer.name] = {
                    "max_abs_diff": abs_diff, "max_rel_diff": _rel(got, want),
                    "scaled_bound": 1e-4 * scale,
                }
                if abs_diff > worst["abs"]:
                    worst = {"layer": layer.name, "abs": abs_diff,
                             "rel": layer_rows[layer.name]["max_rel_diff"],
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
                a = np.concatenate([g.reshape(-1) for g in autograd]).astype(np.float64)
                b = np.concatenate([g.reshape(-1) for g in numeric]).astype(np.float64)
                cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                rel_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(b))
                fd_rows[label] = {"cosine": cosine, "rel_l2_error": rel_l2,
                                  "autograd_norm": float(np.linalg.norm(a)),
                                  "fd_norm": float(np.linalg.norm(b))}
                checks.record(f"{key}/{case_id}/gradient_check/{label}/cosine",
                              cosine >= criteria["gradient_check"]["min_cosine"],
                              cosine, criteria["gradient_check"]["min_cosine"])
                checks.record(f"{key}/{case_id}/gradient_check/{label}/rel_l2",
                              rel_l2 <= criteria["gradient_check"]["max_rel_l2_error"],
                              rel_l2, criteria["gradient_check"]["max_rel_l2_error"])
            case["gradient_check"] = fd_rows

            # ── IG: 사용자 기본 32단계와 정밀 256단계 ──
            handle = rt.explain_handle(runtime, head=head, window_length=window_length,
                                       derive=derive)
            ig_rows = {}
            attributions = {}
            for label, steps in (("default", default["ig_steps"]), ("finer", finer["ig_steps"])):
                result = explain.integrated_gradients(handle, window, steps=steps)
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
                              bool(np.isfinite(result.attribution).all()
                                   and np.isfinite([result.pred, result.baseline_pred]).all()),
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
                      all(w["abs"] <= w["scaled_bound"] for w in layer_worst),
                      worst_layer, criteria["layerwise_parity"]["max_abs_diff_scaled"])
        checks.record(f"{key}/layerwise_parity/rel",
                      all(w["rel"] <= criteria["layerwise_parity"]["max_rel_diff"]
                          for w in layer_worst),
                      max(w["rel"] for w in layer_worst),
                      criteria["layerwise_parity"]["max_rel_diff"])
        per_model["ops_present"] = sorted({
            type(layer).__name__ for layer in runtime._model.layers})
        per_model["layer_names"] = runtime.layer_names()
        per_model["input_names"] = runtime.input_names()
        per_model["head_names"] = runtime.head_names
        measurements[key] = per_model

    report = {
        "schema_version": 1,
        "issue": 193,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "contract_sha256": hashlib.sha256(
            (DOCS / "contract.json").read_bytes()).hexdigest(),
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
runtime.forward(inputs, head=head)
stages["after_forward"] = rss()
from xai import explain
handle = rt.explain_handle(runtime, head=head, window_length=len(window),
                           derive=rt.cas12a_derive if %(key)r == "cas12a" else None)
explain.integrated_gradients(handle, window, steps=%(steps)d)
stages["after_ig"] = rss()
stages["peak_working_set"] = int(process.memory_info().peak_wset)
stages["tensorflow_importable"] = rt.tensorflow_present()
stages["tensorflow_imported"] = "tensorflow" in sys.modules
print("RSSJSON" + json.dumps(stages))
'''


def run_ram(work: Path) -> None:
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
            raise RuntimeError(f"RSS child failed for {key}: {finished.stderr[-2000:]}")
        stages = json.loads(line[0][len("RSSJSON"):])
        stages["elapsed_seconds"] = round(time.time() - started, 1)
        out[key] = stages
    work.mkdir(parents=True, exist_ok=True)
    (work / "ram.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


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
    out: dict = {}

    keras = rt.import_keras()
    for key in ("spcas9", "cas12a"):
        spec = CONTRACT["models"][MODEL_KEYS[key]]
        head = spec["verified_head"]["index"]
        runtime = rt.NonNTRuntime.load(key)
        rows: dict = {}
        kink_layers = [layer for layer in runtime._model.layers
                       if type(layer).__name__ in ("ReLU", "LeakyReLU")
                       or getattr(getattr(layer, "activation", None), "__name__", "") == "relu"]
        probe = keras.Model(runtime._model.inputs,
                            [layer.output for layer in kink_layers]) if kink_layers else None

        for case_id, window in case_windows(key):
            case: dict = {}
            for label, factor in (("x", 1.0), ("alpha0.5", 0.5)):
                point = [np.asarray(a, dtype=np.float32) * factor
                         for a in case_inputs(key, window)]
                mine = runtime.gradients(point, head=head)[1]
                theirs = [reference[f"{key}/{case_id}/grad/{label}/{i}"]
                          for i in range(len(mine))]
                a = np.concatenate([g.reshape(-1) for g in mine]).astype(np.float64)
                b = np.concatenate([g.reshape(-1) for g in theirs]).astype(np.float64)
                max_abs = float(np.max(np.abs(a - b)))
                cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                checks.record(f"{key}/{case_id}/cross_runtime_gradient/{label}/max_abs_diff",
                              max_abs <= parity_criteria["max_abs_diff"], max_abs,
                              parity_criteria["max_abs_diff"])
                checks.record(f"{key}/{case_id}/cross_runtime_gradient/{label}/cosine",
                              cosine >= parity_criteria["min_cosine"], cosine,
                              parity_criteria["min_cosine"])
                case[f"cross_runtime_{label}"] = {
                    "max_abs_diff": max_abs, "cosine": cosine,
                    "torch_norm": float(np.linalg.norm(a)),
                    "tensorflow_norm": float(np.linalg.norm(b)),
                }

            # 방향 미분 vs 방향 차분 (내부 보간점)
            point = [np.asarray(a, dtype=np.float32) * 0.5
                     for a in case_inputs(key, window)]
            grads = runtime.gradients(point, head=head)[1]
            rng = np.random.default_rng(193)
            step = 1e-3
            errors = []
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
            if probe is not None:
                flips = 0
                total = 0
                base_inputs = case_inputs(key, window)
                flat_sizes = [np.asarray(x).size for x in base_inputs]
                h = CONTRACT["criteria"]["gradient_check"]["fd_step"]
                for which, size in enumerate(flat_sizes):
                    for index in range(size):
                        signs = []
                        for sign in (+1, -1):
                            perturbed = [np.array(x, dtype=np.float32, copy=True)
                                         for x in base_inputs]
                            flat = perturbed[which].reshape(-1)
                            flat[index] += sign * h
                            argument = perturbed[0] if len(perturbed) == 1 else perturbed
                            acts = probe(argument, training=False)
                            acts = acts if isinstance(acts, (list, tuple)) else [acts]
                            signs.append(np.concatenate([
                                np.asarray(v.detach().numpy() if hasattr(v, "detach") else v
                                           ).reshape(-1) > 0 for v in acts]))
                        flips += int(np.sum(signs[0] != signs[1]))
                        total += signs[0].size
                case["kink_sign_flips"] = {"flips": flips, "activation_units_probed": total,
                                           "fd_step": h}
            rows[case_id] = case
        out[key] = rows

    report = {
        "schema_version": 1, "issue": 193,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("reference", "verify", "ram", "minimal", "diagnose"))
    parser.add_argument("--work", default=str(WORK))
    args = parser.parse_args()
    work = Path(args.work)
    if args.step == "reference":
        run_reference(work)
        return 0
    if args.step == "verify":
        return run_verify(work)
    if args.step == "ram":
        run_ram(work)
        return 0
    if args.step == "diagnose":
        return run_diagnose(work)
    return run_minimal()


if __name__ == "__main__":
    raise SystemExit(main())
