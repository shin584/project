"""채택한 non-NT 런타임: Keras 3을 **torch backend**로 돌려 전방 추론과 IG 입력
gradient를 같은 런타임에서 계산한다 (#193).

결정 근거와 대안 비교는 `docs/runtime/README.md`, 동결한 수치 계약은
`docs/runtime/contract.json`에 있다. 요점만:

- 두 `.keras`는 Keras 3 보관 형식(config + weights.h5)이라 backend에 묶여 있지 않다.
  변환물을 만들지 않으므로 추적할 변환 SHA도, 변환 도구도 없다.
- gradient를 주지 않는 전방 전용 런타임(ONNX Runtime)을 쓰면 gradient 구현을
  따로 두고 두 런타임의 parity까지 떠안아야 한다. torch autograd로 한 런타임에서 끝낸다.
- torch는 NT 모델 때문에 이미 배포본에 들어 있다. TensorFlow는 배포본에서 계속 제외한다.

**backend는 환경변수가 아니라 이 모듈이 정한다.** 개발자 셸의 `KERAS_BACKEND`가
앱이 실제로 쓰는 backend를 바꾸면 parity 근거가 무의미해지기 때문이다.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import model_source
import numpy as np
from predict import ExplainHandle

#: 이 모듈이 강제하는 Keras backend.
BACKEND = "torch"

#: CPU 스레드 수. 계약에 기록된 값이며 측정 재현에 영향을 준다.
TORCH_THREADS = 4

#: 기본 전방 배치 크기. SHAP은 수천 서열을 한 번에 넘긴다.
BATCH_SIZE = 64

#: 채택 checkpoint. 경로는 `model_source.NON_NT_FILES`의 checksum 고정 항목과 같아야 한다.
CHECKPOINTS: dict[str, str] = {
    "spcas9": "models/non_nt/spcas9-9hydra-v2/model.keras",
    "cas12a": "models/non_nt/cas12a-cnn-20261102/cas12a_cnn_manifest.keras",
}

_keras = None


def import_keras():
    """backend를 고정한 뒤 keras를 import한다. 이미 다른 backend면 거부한다.

    keras는 import 시점에 backend를 확정하므로 순서가 중요하다. 다른 backend로
    이미 올라와 있으면 조용히 섞이는 대신 멈춘다.
    """
    global _keras
    if _keras is not None:
        return _keras
    import sys

    if "keras" in sys.modules:
        existing = sys.modules["keras"].backend.backend()
        if existing != BACKEND:
            raise RuntimeError(
                f"keras가 이미 {existing} backend로 import됐다. non-NT 런타임은 "
                f"{BACKEND} backend를 요구한다 (docs/runtime/README.md)."
            )
    os.environ["KERAS_BACKEND"] = BACKEND
    import keras
    import torch

    if keras.backend.backend() != BACKEND:
        raise RuntimeError(
            f"keras backend가 {keras.backend.backend()}로 올라왔다. {BACKEND}가 필요하다."
        )
    torch.set_num_threads(TORCH_THREADS)
    _keras = keras
    return _keras


def runtime_versions() -> dict[str, str]:
    """계약에 적어 둔 버전과 실제 실행 버전을 대조할 수 있게 한다."""
    keras = import_keras()
    import torch
    import platform
    import h5py
    from importlib.metadata import version

    return {
        "python": platform.python_version(),
        "keras": keras.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "shap": version("shap"),
        "h5py": h5py.__version__,
        "backend": keras.backend.backend(),
        "torch_threads": str(torch.get_num_threads()),
    }


def tensorflow_present() -> bool:
    """TensorFlow가 이 프로세스에서 import 가능한지.

    `Bidirectional` 역직렬화가 `tf.available`을 묻기 때문에, 설치돼 있으면
    torch backend에서도 TensorFlow가 sys.modules에 올라온다. 의존이 아니라
    탐지일 뿐이지만, 측정값에 섞이면 RAM과 배포 크기를 과대평가하게 된다.
    배포본은 TensorFlow를 제외하므로 이 값이 False인 환경이 기준이다.
    """
    import importlib.util

    return importlib.util.find_spec("tensorflow") is not None


class NonNTRuntime:
    """채택 checkpoint 하나에 대한 전방·gradient 접근.

    한 프로세스에 선택한 계열의 모델만 올린다 (#191). 다른 계열을 쓰려면
    프로세스를 교체한다 — 이 객체는 교체 대상이지 전환 대상이 아니다.
    """

    def __init__(self, key: str, model) -> None:
        self.key = key
        self._model = model
        self._probe = None
        self._probe_names: list[str] = []

    @classmethod
    def load(cls, key: str) -> NonNTRuntime:
        """checksum이 고정된 경로에서 모델을 적재한다.

        `model_source.resolve`가 크기·SHA-256을 확인하므로, 바뀐 파일로는
        추론하지 않는다 (#192의 보관 계약).

        **캐시하지 않는다.** 적재한 모델의 수명은 그 프로세스를 가진 쪽이 정한다
        (#194의 계산 프로세스 교체). 여기서 모듈 수준 캐시를 두면 교체로 메모리를
        되찾는다는 설계와 어긋나고, CONTEXT.md가 세는 캐시 목록에도 없는 네 번째
        캐시가 생긴다.
        """
        if key not in CHECKPOINTS:
            raise KeyError(
                f"알 수 없는 non-NT checkpoint: {key!r}. {sorted(CHECKPOINTS)} 중 하나여야 한다."
            )
        keras = import_keras()
        path = model_source.resolve(CHECKPOINTS[key])
        return cls(key, keras.saving.load_model(path, compile=False))

    # ── 구조 ──────────────────────────────────────────────

    @property
    def input_shapes(self) -> list[tuple[int, ...]]:
        """배치 차원을 뺀 입력 모양 목록."""
        return [tuple(int(d) for d in t.shape[1:]) for t in self._model.inputs]

    @property
    def n_outputs(self) -> int:
        return len(self._model.outputs)

    @property
    def head_names(self) -> list[str]:
        """설명 대상이 되는 출력 이름. SpCas9은 9개, Cas12a는 1개다."""
        return list(self._model.output_names)

    def input_names(self) -> list[str]:
        return [tensor.name for tensor in self._model.inputs]

    def layer_names(self) -> list[str]:
        return [layer.name for layer in self._model.layers]

    def ops_present(self) -> list[str]:
        """이 모델이 실제로 쓰는 연산 종류. parity 근거에 무엇이 포함됐는지 적는다."""
        return sorted({type(layer).__name__ for layer in self._model.layers})

    def weight_digests(self) -> dict[str, dict]:
        """변수별 모양·dtype·float32 바이트의 SHA-256.

        backend를 바꿔 적재했을 때 가중치가 재해석·전치되지 않았는지 대조하는
        용도다 (#193의 weight_placement 기준). 적재에는 산술이 없으므로 비트
        동일성이 맞는 기준이다.
        """
        import hashlib

        digests = {}
        for variable in self._model.weights:
            value = np.ascontiguousarray(np.asarray(variable.numpy(), dtype=np.float32))
            digests[variable.path] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        return digests

    def nonsmooth_layer_names(self) -> list[str]:
        """0에서 꺾이는 활성(ReLU·LeakyReLU)을 내보내는 층 이름.

        좌표 차분으로 gradient를 검증할 때, 섭동이 이 꺾임을 넘으면 차분 몫이
        한쪽 기울기도 다른 쪽 기울기도 아닌 값이 된다. 실패를 설명하려면 몇 번
        넘었는지 세야 하고, 그러려면 어느 층을 봐야 하는지 알아야 한다.

        이름으로 찾으므로, 다른 이름을 쓰는 꺾인 활성이 들어오면 빈 목록을
        돌려준다. **진단용으로만** 쓴다 — 어떤 판정도 이 목록에 기대지 않는다.
        """
        names = []
        for layer in self._model.layers:
            activation = getattr(layer, "activation", None)
            if (type(layer).__name__ in ("ReLU", "LeakyReLU")
                    or getattr(activation, "__name__", "") == "relu"):
                names.append(layer.name)
        return names

    def _activation_probe(self):
        """모든 층의 출력을 한 번에 내보내는 모델. 한 번만 만들어 재사용한다.

        호출마다 다시 만들면 꺾임 세기처럼 수백 번 부르는 검증에서 그 비용이
        전부 그래프 재구성에 들어간다. 적재한 모델에서 파생된 그래프일 뿐이고
        모델 캐시가 아니므로, `load`가 캐시하지 않는다는 규칙과 무관하다.
        """
        if self._probe is None:
            keras = import_keras()
            inner = [layer for layer in self._model.layers
                     if not isinstance(layer, keras.layers.InputLayer)]
            self._probe_names = [layer.name for layer in inner]
            self._probe = keras.Model(self._model.inputs,
                                       [layer.output for layer in inner])
        return self._probe, self._probe_names

    def layer_activations(self, arrays) -> dict[str, np.ndarray]:
        """층 이름 → 그 층의 추론 모드 출력.

        최종 스칼라만 비교하면 전치된 커널이나 뒤바뀐 LSTM gate 순서가 숨을 수
        있다. 층별로 비교해야 '같은 연산'이라는 주장에 근거가 생긴다.
        """
        probe, names = self._activation_probe()
        outputs = probe(self._as_argument(self._check(arrays)), training=False)
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        return {name: self._to_numpy(value) for name, value in zip(names, outputs)}

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        return np.asarray(value.detach().numpy() if hasattr(value, "detach") else value)

    # ── 전방 ──────────────────────────────────────────────

    def _as_argument(self, tensors):
        """Keras는 입력이 하나면 목록이 아니라 그 자체를 받는다."""
        return tensors[0] if len(tensors) == 1 else list(tensors)

    def _call(self, tensors):
        """추론 모드 호출. dropout은 꺼지고 BatchNormalization은 이동통계를 쓴다."""
        return self._model(self._as_argument(tensors), training=False)

    def _head(self, out, head: int):
        if isinstance(out, (list, tuple)):
            if not 0 <= head < len(out):
                raise IndexError(f"head {head}는 출력 {len(out)}개 범위 밖이다")
            picked = out[head]
        else:
            if head != 0:
                raise IndexError(f"이 모델은 출력이 1개인데 head {head}를 요구했다")
            picked = out
        return picked.reshape(-1)

    def forward(self, arrays, head: int = 0, batch_size: int = BATCH_SIZE,
                on_batch: Callable | None = None) -> np.ndarray:
        """입력 배열 목록(각각 [N, ...]) → 선택한 head의 원출력 [N].

        on_batch(done, total)은 배치 완료마다 **서열 수**를 보고한다. 콜백 예외는
        호출자에게 전파해 계산을 취소할 수 있다 (ADR 0001).
        배치 크기는 값에 영향을 주지 않는다.
        """
        import torch

        arrays = [np.asarray(a, dtype=np.float32) for a in self._check(arrays)]
        total = arrays[0].shape[0]
        out = np.empty(total, dtype=np.float32)
        step = max(1, int(batch_size))
        for start in range(0, total, step):
            stop = min(start + step, total)
            with torch.no_grad():
                tensors = [torch.from_numpy(a[start:stop]) for a in arrays]
                y = self._head(self._call(tensors), head)
            out[start:stop] = y.detach().numpy()
            if on_batch is not None:
                on_batch(stop, total)
        return out

    # ── gradient ─────────────────────────────────────────

    def gradients(self, arrays, head: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
        """입력 배열 목록 → (선택 head의 출력 [N], 입력별 기울기 목록).

        배치 행은 서로 독립이므로 합의 기울기가 행별 기울기와 같다. explain.py의
        IG 루프가 이 성질에 의존한다.
        """
        import torch

        arrays = [np.asarray(a, dtype=np.float32) for a in self._check(arrays)]
        tensors = [torch.from_numpy(a.copy()).requires_grad_(True) for a in arrays]
        y = self._head(self._call(tensors), head)
        y.sum().backward()
        grads = []
        for name, tensor in zip(self.input_names(), tensors):
            if tensor.grad is None:
                raise RuntimeError(f"입력 {name}에 기울기가 흐르지 않았다")
            grads.append(tensor.grad.detach().numpy().copy())
        return y.detach().numpy().copy(), grads

    def _check(self, arrays):
        arrays = list(arrays)
        shapes = self.input_shapes
        if len(arrays) != len(shapes):
            raise ValueError(
                f"{self.key}는 입력 {len(shapes)}개를 받는다 ({shapes}), {len(arrays)}개가 왔다"
            )
        counts = {np.asarray(a).shape[0] for a in arrays}
        if len(counts) != 1:
            raise ValueError(f"입력들의 배치 크기가 다르다: {counts}")
        for array, shape in zip(arrays, shapes):
            got = tuple(np.asarray(array).shape[1:])
            if got != shape:
                raise ValueError(f"{self.key} 입력 모양이 {shape}가 아니라 {got}다")
        return arrays


# ════════════════════════════════════════════════
# 기존 XAI 루프에 꽂는 handle
# ════════════════════════════════════════════════

ONE_HOT_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(seq: str) -> np.ndarray:
    """ACGT 외 문자는 전부 0으로 둔다 (xai/explain.py의 `_onehot`과 같은 규약)."""
    out = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in ONE_HOT_INDEX:
            out[i, ONE_HOT_INDEX[base]] = 1.0
    return out


@dataclass
class RuntimeHandle(ExplainHandle):
    """`ExplainHandle`에 입력별 기여도 합계만 더한 것.

    표면을 다시 선언하지 않고 상속한다. handle 표면이 바뀔 때 두 곳을 고쳐야 하는
    상태를 만들지 않기 위해서다. `head`는 상위 계약대로 **출력 이름**이고,
    `NonNTRuntime`의 메서드가 받는 `head`는 그 출력의 인덱스다.

    `project`는 **서열 입력**의 위치별 기여도만 돌려준다. 파생 특징 입력이 있는
    Cas12a에서 그 기여도를 염기에 임의 배분하지 않기 위해서다. 대신 입력별 signed
    합계를 `input_totals`에 남겨, 호출자가 특징 쪽 기여를 모른 채 잃지 않게 한다.
    두 입력을 화면에 어떻게 제시할지, 그리고 합성 경로 `F(x) = model(x, derive(x))`를
    채택할지는 #195가 정한다. 이 handle은 그 결정을 대신하지 않는다.
    """

    #: 입력 이름 → 마지막 `project` 호출에서 본 signed 기여도 합. `project`가
    #: 제자리에서 갱신하므로, 호출자가 들고 있는 것과 같은 사전이다.
    input_totals: dict[str, float] = field(default_factory=dict)


def explain_handle(runtime: NonNTRuntime, head: int, window_length: int,
                   derive: Callable | None = None,
                   model_key: str | None = None) -> RuntimeHandle:
    """런타임 하나를 기존 IG/SHAP 루프에 쓸 수 있는 handle로 감싼다.

    derive(seq) -> [F] 를 주면 파생 특징 입력이 있는 모델(Cas12a)이 되고, IG는
    현재 구현된 **이중 입력** 경로로 두 입력을 각자의 영벡터에서 보간한다.
    SHAP이 섞어 만든 원핫은 서열로 되돌리지 않고 그대로 넣되, 파생 특징은
    그 원핫에서 다시 계산한다 (#191: 서열을 바꾸면 파생 특징도 함께 바뀐다).
    """
    handle_head_name = runtime.head_names[head]

    def _features_for(X: np.ndarray) -> np.ndarray:
        return np.stack([derive(_onehot_to_seq(row)) for row in X]).astype(np.float32)

    def raw_fn(X, on_batch: Callable | None = None):
        X = np.asarray(X, dtype=np.float32)
        arrays = [X] if derive is None else [X, _features_for(X)]
        return runtime.forward(arrays, head=head, on_batch=on_batch)

    def ig_inputs(seq: str):
        seq_input = one_hot(seq)
        if derive is None:
            return [seq_input]
        return [seq_input, np.asarray(derive(seq), dtype=np.float32)]

    def ig_grad_fn(batch):
        return runtime.gradients(list(batch), head=head)

    input_totals: dict[str, float] = {}

    def project(attrs):
        attrs = [np.asarray(a, dtype=np.float64) for a in attrs]
        input_totals.clear()
        input_totals.update(
            {name: float(a.sum()) for name, a in zip(runtime.input_names(), attrs)}
        )
        positions = attrs[0].sum(axis=-1)
        if positions.shape[0] != window_length:
            raise ValueError(
                f"윈도우 길이가 {window_length}인데 기여도는 {positions.shape[0]}개다"
            )
        return positions

    return RuntimeHandle(
        raw_fn=raw_fn, ig_inputs=ig_inputs, ig_grad_fn=ig_grad_fn, project=project,
        attention_fn=None, model_key=model_key, head=handle_head_name,
        input_totals=input_totals,
    )


def cas12a_derive(seq: str) -> np.ndarray:
    """Cas12a의 파생 특징 354개 = GC 2개 뒤에 위치별 dinucleotide 352개.

    이 **순서**가 학습 때와 같아야 한다. 학습 코드
    (`models/non_nt/cas12a-cnn-20261102/source_snapshot.zip`의 `cas12a/data.py`)는
    `np.concatenate([gc_feat, dinuc_feat])`로 만들었고, 그 snapshot의
    `cas12a/features.py`는 이 워크트리의 것과 줄바꿈만 다르고 내용이 같다.
    순서가 틀리면 예측은 그럴듯하게 나오면서 조용히 틀린다 (#91).

    여기서는 문자열을 세는 기존 추출기를 그대로 쓴다. 미분 가능한 동등 구현이
    필요한 합성 IG 경로는 #195가 정한다.
    """
    from cas12a import features

    gc_feat, dinuc_feat = features.extract_seq_features(seq)
    return np.concatenate([gc_feat, dinuc_feat]).astype(np.float32)


def _onehot_to_seq(row: np.ndarray) -> str:
    """원핫 한 줄 → 서열. 전부 0인 위치는 'N'으로 둔다."""
    bases = "ACGT"
    out = []
    for position in np.asarray(row):
        if float(position.sum()) <= 0:
            out.append("N")
        else:
            out.append(bases[int(np.argmax(position))])
    return "".join(out)
