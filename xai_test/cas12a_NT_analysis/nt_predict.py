"""Native scalar raw prediction in one explicitly recorded FP32 execution path."""

import hashlib
import importlib.metadata
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from cas12a.nt_data import digest

from .nt_inputs import (
    MAX_LENGTH,
    check_export_files,
    encode_sequences,
    validate_vocabulary,
)
from .nt_run import file_hash

ATOL = 1e-5
RTOL = 1e-4


def installed_environment(device, *, query_gpu=False):
    import torch

    packages = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device)
        if query_gpu and str(device).startswith("cuda")
        else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if query_gpu else None,
    }


def source_snapshot():
    """Copy the complete analysis and its read-only Cas12a import closure."""
    project = Path(__file__).resolve().parents[2]
    analysis = Path(__file__).resolve().parent
    paths = sorted(analysis.glob("*.py"))
    paths += [
        project / "cas12a" / name
        for name in (
            "__init__.py",
            "cnn_manifest.py",
            "nt_data.py",
            "data.py",
            "features.py",
            "units.py",
        )
    ]
    paths.append(analysis / "docs" / "planning" / "issue139-execution-contract.md")
    paths.append(analysis / "docs" / "planning" / "issue141-ism-contract.md")
    paths.append(analysis / "docs" / "planning" / "issue142-ig-contract.md")
    paths.append(analysis / "docs" / "planning" / "issue143-rollout-contract.md")
    paths.append(analysis / "docs" / "planning" / "issue146-colab-contract.md")
    paths.append(analysis / "docs" / "reports" / "issue146-design-review.md")
    return {path.relative_to(project).as_posix(): path.read_bytes() for path in paths}


def state_digest(model, budget_guard=None):
    import torch

    checksum = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if budget_guard:
            budget_guard()
        checksum.update(name.encode())
        checksum.update(str(tensor.dtype).encode())
        checksum.update(str(tuple(tensor.shape)).encode())
        checksum.update(
            tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        )
        if budget_guard:
            budget_guard()
    return checksum.hexdigest()


@contextmanager
def fp32_context(device):
    import torch

    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device.type, enabled=False), torch.no_grad():
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class NativePredictor:
    def __init__(
        self, model, tokenizer, export_dir, *, budget_guard=None, export_files=None
    ):
        self.model, self.tokenizer, self.export_dir = model, tokenizer, Path(export_dir)
        self.embedding_rows = model.get_input_embeddings().weight.shape[0]
        self.vocabulary = validate_vocabulary(tokenizer, self.embedding_rows)
        self.budget_guard = budget_guard or (lambda: None)
        self.export_files = export_files
        self.initial_state = state_digest(model, self.budget_guard)
        self.actual_execution = None

    @classmethod
    def load(cls, export_dir, *, device="cpu", budget_guard=None, expected_files=None):
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        guard = budget_guard or (lambda: None)
        guard()
        export_dir = Path(export_dir)
        expected_files = expected_files or {
            path.relative_to(export_dir).as_posix(): file_hash(path)
            for path in export_dir.rglob("*")
            if path.is_file()
        }
        check_export_files(export_dir, expected_files)
        guard()
        config = AutoConfig.from_pretrained(
            export_dir, local_files_only=True, trust_remote_code=False
        )
        if (
            config.model_type != "esm"
            or config.num_labels != 1
            or config.problem_type != "regression"
        ):
            raise ValueError("saved config is not the native scalar ESM regressor")
        tokenizer = AutoTokenizer.from_pretrained(
            export_dir, local_files_only=True, trust_remote_code=False
        )
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            export_dir,
            config=config,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if any(
            loading.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"incomplete saved model reload: {loading}")
        guard()
        model = model.eval().to(device=device, dtype=torch.float32)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        guard()
        check_export_files(export_dir, expected_files)
        guard()
        return cls(
            model,
            tokenizer,
            export_dir,
            budget_guard=guard,
            export_files=expected_files,
        )

    def set_budget_guard(self, guard):
        self.budget_guard = guard

    def synchronize(self):
        import torch

        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def assert_unchanged(self):
        if state_digest(self.model, self.budget_guard) != self.initial_state:
            raise ValueError("model weights/buffers changed during inference")

    def _batch(self, sequences):
        import torch

        self.budget_guard()
        if any(module.training for module in self.model.modules()):
            raise ValueError("native prediction requires eval mode")
        if any(
            parameter.dtype != torch.float32 for parameter in self.model.parameters()
        ):
            raise ValueError("native prediction requires FP32 weights")
        if self.model.config._attn_implementation != "eager":
            raise ValueError("attention backend must be eager")
        records = encode_sequences(self.tokenizer, sequences, self.embedding_rows)
        device = next(self.model.parameters()).device
        encoded = {
            key: torch.tensor(
                [row[key] for row in records], dtype=torch.long, device=device
            )
            for key in ("input_ids", "attention_mask")
        }
        self.synchronize()
        self.budget_guard()
        with fp32_context(device):
            logits = self.model(**encoded).logits
            self.synchronize()
            self.budget_guard()
            if logits.dtype != torch.float32 or tuple(logits.shape) != (
                len(sequences),
                1,
            ):
                raise ValueError("expected scalar FP32 native raw output")
            self.actual_execution = {
                "parameter_dtype": "torch.float32",
                "output_dtype": str(logits.dtype),
                "autocast": torch.is_autocast_enabled(device.type),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "attention_backend": self.model.config._attn_implementation,
                "eval": not self.model.training,
                "gradient_enabled": torch.is_grad_enabled(),
                "max_length": MAX_LENGTH,
                "device": str(device),
            }
            values = logits[:, 0].detach().cpu().numpy()
            self.budget_guard()
        if not np.isfinite(values).all():
            raise ValueError("nonfinite native prediction")
        return values

    def predict(self, sequences, *, batch_size=32):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch size must be a positive integer")
        outputs = [
            self._batch(sequences[start : start + batch_size])
            for start in range(0, len(sequences), batch_size)
        ]
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)

    def parity(self, sequences, *, batch_size=32):
        if not sequences:
            raise ValueError("parity requires validation samples")
        native = self._batch(sequences)
        adapter = self.predict(sequences, batch_size=batch_size)
        singleton = self.predict(sequences, batch_size=1)
        reloaded = type(self).load(
            self.export_dir,
            device=str(next(self.model.parameters()).device),
            budget_guard=self.budget_guard,
            expected_files=self.export_files,
        )
        fresh = reloaded.predict(sequences, batch_size=batch_size)
        comparisons = {
            "adapter": adapter,
            "singleton": singleton,
            "fresh_reload": fresh,
        }
        for name, values in comparisons.items():
            if not np.allclose(values, native, atol=ATOL, rtol=RTOL):
                raise ValueError(f"FP32 parity failure: {name}")
        self.assert_unchanged()
        reloaded.assert_unchanged()
        if reloaded.initial_state != self.initial_state:
            raise ValueError("fresh reload weight identity mismatch")
        return {
            "validated": True,
            "atol": ATOL,
            "rtol": RTOL,
            "samples": len(sequences),
            "max_absolute_difference": {
                name: float(np.max(np.abs(values - native)))
                for name, values in comparisons.items()
            },
            "state_sha256": self.initial_state,
            "vocabulary": self.vocabulary,
            "actual_execution": self.actual_execution,
        }


def execution_identity(inputs, environment, snapshots, batch_size):
    from .nt_ig import PROTOCOL as IG_PROTOCOL
    from .nt_ism import PROTOCOL as ISM_PROTOCOL
    from .nt_rollout import PROTOCOL as ROLLOUT_PROTOCOL

    return {
        **inputs.identity,
        "code_sha256": digest(
            {name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()}
        ),
        "packages": environment["packages"],
        "python": environment["python"],
        "protocol": {
            "schema": 1,
            "precision": "float32",
            "autocast": False,
            "tf32": False,
            "backend": "eager",
            "atol": ATOL,
            "rtol": RTOL,
            "batch_size": batch_size,
            "ism": ISM_PROTOCOL,
            "ig": IG_PROTOCOL,
            "rollout": ROLLOUT_PROTOCOL,
        },
    }
