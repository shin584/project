"""Thin read-only access to the NT XAI 데모's prepared artifacts (`data/`).

The reader presents figures the analyses already checked. It never recomputes
or re-validates an analysis, never loads a model (no torch/transformers) and
does not import the analysis packages (`cas12a_NT_analysis`,
`SpCas9_NT_analysis`): their import styles are incompatible and importing them
would spread this subtree's test scope.

The price is that no existing check catches a misread here. The manifest only
detects changed files, not a correct file read wrongly, so the four misreads
that are easy to make on these artifacts are closed by structure and covered by
the synthetic fixture tests:

* **Heads by name.** SpCas9 ISM `delta` has a 9-head axis (`heads`, of which
  only `analysed_heads` are analysed) and IG has a 3-head axis in its own
  order. A head is resolved to an axis position through each artifact's own
  sidecar, never by position; only analysed heads are accepted.
* **Rows by ID.** The only axis the artifacts share is `row_index` (Cas12a:
  `source_row`). Every lookup goes through that ID, never through a position.
* **Units.** SpCas9 split label columns are percent; predictions and ISM deltas
  are fractions; IG values and completeness residuals are on the standardized
  output. Labels are returned as fractions, and every scale an artifact
  declares is checked on load instead of assumed.
* **WT slot.** An ISM position profile is the mean |delta| over the three
  alternative bases; the reference (WT) slot, a true zero, is excluded.

The profile definitions here are the ones the sidecars state, and on the real
data they reproduce the analyses' own recorded profiles exactly
(`test_reader_real_data.py`).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
SPLITS = ("train", "val", "test")
BASES = ("A", "C", "G", "T")

# Cas12a has one regression output, so there is no head axis to choose from.
# Naming it anyway keeps every displayed figure carrying a head field.
CAS12A_HEAD = "단일 출력"
CAS12A_STRATA = ("all", "TTTV", "TTTT")
CAS12A_LIMIT_SECTIONS = ("rollout 해석", "범위와 한계")


class ArtifactMismatch(RuntimeError):
    """An artifact does not declare what this reader was written against."""


class UnknownHead(KeyError):
    """A head name is not on the artifact's head axis, or is not analysed."""


class RowNotPresent(KeyError):
    """A row ID is not in this artifact's population."""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def expect(declared: Any, wanted: Any, *, what: str, source: Path | str) -> None:
    """Fail loudly when an artifact's own declaration is not what we read it as."""
    if declared != wanted:
        raise ArtifactMismatch(
            f"{source}: {what}이(가) {wanted!r}가 아니라 {declared!r}이다"
        )


def percent_to_fraction(value: float) -> float:
    """Split label columns are percent; predictions and ISM deltas are fractions."""
    return value / 100.0


@dataclass(frozen=True)
class HeadAxis:
    """The head names along one artifact's head axis, in that artifact's order.

    `names` is the stored axis (SpCas9 ISM stores all nine heads from the one
    forward pass); `analysed` is the subset any result covers. A head is looked
    up by name, so a stored-but-not-analysed head and a different axis order
    both fail instead of silently returning another head's numbers.
    """

    names: tuple[str, ...]
    analysed: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ArtifactMismatch(f"{self.source}: head 이름이 중복된다")
        missing = [head for head in self.analysed if head not in self.names]
        if missing:
            raise ArtifactMismatch(
                f"{self.source}: analysed_heads {missing}가 head 축에 없다"
            )

    @property
    def width(self) -> int:
        return len(self.names)

    def index(self, head: str) -> int:
        if head not in self.names:
            raise UnknownHead(
                f"{self.source}: head {head!r}가 없다 (있는 head: {', '.join(self.names)})"
            )
        if head not in self.analysed:
            raise UnknownHead(
                f"{self.source}: head {head!r}는 저장만 되고 분석되지 않았다 "
                f"(분석된 head: {', '.join(self.analysed)})"
            )
        return self.names.index(head)


# `eq=False`: the generated `__eq__`/`__hash__` would compare and hash the ID
# array, which raises rather than answering. Axes are identified by the
# artifact they came from, not compared.
@dataclass(frozen=True, eq=False)
class RowAxis:
    """An artifact's row IDs, the only identity shared between artifacts."""

    ids: np.ndarray
    id_name: str
    source: str

    @cached_property
    def _position(self) -> dict[int, int]:
        return {int(row_id): index for index, row_id in enumerate(self.ids)}

    @property
    def rows(self) -> int:
        return len(self.ids)

    def __contains__(self, row_id: int) -> bool:
        return int(row_id) in self._position

    def position(self, row_id: int) -> int:
        try:
            return self._position[int(row_id)]
        except KeyError:
            raise RowNotPresent(
                f"{self.source}: {self.id_name} {row_id}은(는) 이 {self.rows}행 모집단에 없다"
            ) from None

    def positions(self, row_ids: Iterable[int]) -> np.ndarray:
        return np.array([self.position(row_id) for row_id in row_ids], dtype=np.int64)


def reference_mask(reference_base: np.ndarray, bases: int = len(BASES)) -> np.ndarray:
    """One-hot of the reference (WT) base per row and position."""
    return np.eye(bases, dtype=bool)[reference_base]


def ism_position_profile(
    delta: np.ndarray,
    reference_base: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Mean |delta| over the alternative bases, with the WT slot excluded.

    `delta` is `(..., positions, bases)` for one head already selected and
    `reference_base` is `(..., positions)`. The reference slot holds a true
    zero - substituting a base for itself changes nothing - so averaging over
    all four bases would divide a three-base sum by four and shrink every
    position by a quarter. The divisor is the count of slots actually averaged,
    and a position with no valid alternative yields NaN rather than a zero that
    would read as "no effect".
    """
    delta = np.asarray(delta)
    reference_base = np.asarray(reference_base)
    if delta.shape[:-1] != reference_base.shape:
        raise ArtifactMismatch(
            f"delta {delta.shape}와 reference_base {reference_base.shape}의 축이 맞지 않는다 "
            "(head 축을 먼저 이름으로 고르라)"
        )
    is_reference = reference_mask(reference_base, delta.shape[-1])
    on_reference = np.abs(delta)[is_reference]
    if on_reference.size and np.max(on_reference) != 0.0:
        raise ArtifactMismatch(
            "reference(WT) 슬롯이 0이 아니다: 축 해석이 틀렸을 수 있다"
        )
    alternatives = ~is_reference
    if valid is not None:
        alternatives &= np.asarray(valid, dtype=bool)
    divisor = alternatives.sum(axis=-1)
    total = np.where(alternatives, np.abs(delta), 0.0).sum(axis=-1)
    return np.divide(
        total,
        divisor,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=divisor > 0,
    )


def aggregate_to_tokens(
    profile: np.ndarray, boundaries: Sequence[Sequence[int]]
) -> np.ndarray:
    """Mean of a per-position profile over each token's inclusive position span.

    IG attributes to tokens and ISM substitutes bases, so a comparison between
    them is made at the coarser of the two resolutions (ADR 0007). This exists
    to put the reader's own ISM profile next to a *recorded* token profile and
    check they agree; the demo presents the recorded figures, not these.
    """
    profile = np.asarray(profile)
    return np.stack(
        [profile[..., start : end + 1].mean(axis=-1) for start, end in boundaries],
        axis=-1,
    )


# --------------------------------------------------------------------------- #
# SpCas9 (9-Hydra v3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IgAbsence:
    """Why a row carries no IG result. `explain` states this instead of a gap."""

    row_id: int
    rows: int
    reason: str


class SpCas9Reader:
    """Reads the prepared 9-Hydra v3 artifacts. Nothing here recomputes a result."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.root = Path(data_dir) / "spcas9"
        if not self.root.is_dir():
            raise FileNotFoundError(f"SpCas9 자료가 없다: {self.root}")

    # -- sidecars and declared scales --------------------------------------- #

    @cached_property
    def ism_sidecar(self) -> dict[str, Any]:
        path = self.root / "v3_ism/ism.sidecar.json"
        sidecar = load_json(path)
        expect(sidecar["delta_scale"], "fraction", what="delta_scale", source=path)
        expect(list(sidecar["bases"]), list(BASES), what="bases", source=path)
        expect(sidecar["nan_policy"], "forbidden", what="nan_policy", source=path)
        return sidecar

    @cached_property
    def ig_sidecar(self) -> dict[str, Any]:
        path = self.root / "v3_ig/ig.sidecar.json"
        sidecar = load_json(path)
        expect(sidecar["token_resolution"], 5, what="token_resolution", source=path)
        # IG declares no scale key; what fixes the scale of its values is where
        # the gradients were taken and against which baseline. Pinning both is
        # how `ig_scale` below can state the basis instead of assuming it.
        expect(
            sidecar["attribution_point"],
            "encoder_input_embedding",
            what="attribution_point",
            source=path,
        )
        expect(sidecar["baseline"], "zero_embedding", what="baseline", source=path)
        return sidecar

    @property
    def ig_scale(self) -> str:
        """IG values and the completeness residual are on the standardized output.

        Not fractions: the head's standardized regression output, which is what
        `f_baseline` and the completeness residual are measured against. A view
        tags IG figures with this so they are never read next to a prediction
        as if both were efficiencies.
        """
        return "표준화 출력 (fraction 아님)"

    @cached_property
    def predictions_checksum(self) -> dict[str, Any]:
        path = self.root / "v3_testset_predictions/predictions_checksum.json"
        checksum = load_json(path)
        expect(
            checksum["prediction_scale"],
            "fraction",
            what="prediction_scale",
            source=path,
        )
        expect(
            checksum["label_scale_in_split"],
            "percent",
            what="label_scale_in_split",
            source=path,
        )
        return checksum

    @cached_property
    def validation(self) -> dict[str, Any]:
        path = self.root / "v3_report/ig_ism_validation.json"
        report = load_json(path)
        expect(report["pooling"], "forbidden", what="pooling", source=path)
        return report

    @cached_property
    def noise_floor(self) -> dict[str, Any]:
        return load_json(self.root / "v3_noise_floor.json")

    @cached_property
    def head_order_reversal(self) -> dict[str, Any]:
        return load_json(self.root / "v3_report/head_order_reversal.json")

    @cached_property
    def head_agreement(self) -> dict[str, Any]:
        return load_json(self.root / "v3_head_agreement/head_agreement.json")

    @cached_property
    def ig_subset(self) -> dict[str, Any]:
        return load_json(self.root / "v3_ig_subset/ig_subset.json")

    @cached_property
    def cases(self) -> list[dict[str, Any]]:
        """The 18 selected cases, keeping their own `case_id` (GEN-ERRHI-01 ...)."""
        return list(load_json(self.root / "v3_cases/cases.json")["cases"])

    @cached_property
    def cases_annotated(self) -> dict[str, Any]:
        return load_json(self.root / "v3_cases/cases_annotated.json")

    # -- axes --------------------------------------------------------------- #

    @cached_property
    def ism_heads(self) -> HeadAxis:
        sidecar = self.ism_sidecar
        return HeadAxis(
            names=tuple(sidecar["heads"]),
            analysed=tuple(sidecar["analysed_heads"]),
            source="SpCas9 ISM (9축)",
        )

    @cached_property
    def ig_heads(self) -> HeadAxis:
        heads = tuple(self.ig_sidecar["heads"])
        return HeadAxis(names=heads, analysed=heads, source="SpCas9 IG (3축)")

    @property
    def primary_head(self) -> str:
        return str(self.ig_subset["primary_head"])

    @property
    def analysed_heads(self) -> tuple[str, ...]:
        return self.ism_heads.analysed

    @property
    def token_names(self) -> tuple[str, ...]:
        return tuple(self.ig_sidecar["token_names"])

    @property
    def token_boundaries(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(start), int(end)) for start, end in self.ig_sidecar["token_boundaries"]
        )

    @property
    def strata(self) -> tuple[str, ...]:
        return tuple(self.validation["strata"])

    def require_stratum(self, stratum: str) -> str:
        """A stratum has to be one the artifact declares, and only one of them."""
        if stratum not in self.strata:
            raise KeyError(
                f"SpCas9 층 {stratum!r}이 없다 (있는 층: {', '.join(self.strata)})"
            )
        return stratum

    @cached_property
    def prediction_heads(self) -> HeadAxis:
        """The head axis of the split and prediction tables, which store all nine.

        Separate from `ism_heads` on purpose: those tables are a different
        artifact with its own declared head list, and this demo exposes only
        the analysed three (#155).
        """
        return HeadAxis(
            names=tuple(self.predictions_checksum["heads"]),
            analysed=self.ism_heads.analysed,
            source="SpCas9 예측·split 표 (9열)",
        )

    # -- arrays ------------------------------------------------------------- #

    @cached_property
    def _ism(self) -> dict[str, np.ndarray]:
        with np.load(self.root / "v3_ism/ism.npz") as archive:
            arrays = {
                key: archive[key]
                for key in ("delta", "wt_prediction", "reference_base", "row_index")
            }
        heads = self.ism_heads.width
        if arrays["delta"].shape[-1] != heads:
            raise ArtifactMismatch(
                f"ISM delta의 head 축이 {arrays['delta'].shape[-1]}인데 sidecar는 {heads}개를 선언한다"
            )
        expect(
            len(arrays["row_index"]),
            self.ism_sidecar["rows"],
            what="ISM 행 수",
            source="ism.npz",
        )
        return arrays

    @cached_property
    def _ig(self) -> dict[str, np.ndarray]:
        with np.load(self.root / "v3_ig/ig.npz") as archive:
            arrays = {
                key: archive[key]
                for key in ("attribution", "completeness_residual", "row_index")
            }
        if arrays["attribution"].shape[1] != self.ig_heads.width:
            raise ArtifactMismatch(
                f"IG attribution의 head 축이 {arrays['attribution'].shape[1]}인데 "
                f"sidecar는 {self.ig_heads.width}개를 선언한다"
            )
        expect(
            len(arrays["row_index"]),
            self.ig_sidecar["rows"],
            what="IG 행 수",
            source="ig.npz",
        )
        return arrays

    @cached_property
    def _rollout(self) -> dict[str, np.ndarray]:
        with np.load(self.root / "v3_rollout/rollout.npz") as archive:
            return {key: archive[key] for key in ("rollout", "row_index")}

    @cached_property
    def ism_rows(self) -> RowAxis:
        return RowAxis(self._ism["row_index"], "row_index", "SpCas9 ISM 모집단")

    @cached_property
    def ig_rows(self) -> RowAxis:
        return RowAxis(self._ig["row_index"], "row_index", "SpCas9 IG 고정 명단")

    @cached_property
    def rollout_rows(self) -> RowAxis:
        return RowAxis(self._rollout["row_index"], "row_index", "SpCas9 rollout 모집단")

    # -- split and predictions ---------------------------------------------- #

    @cached_property
    def _split(self) -> dict[int, tuple[str, dict[str, float | None]]]:
        """row_index -> (split role, labels as fractions, None where unmeasured).

        The label columns are percent in the split files; they are divided here
        so no caller has to remember which scale it is holding. A blank cell is
        a head that was not measured on that row - 165 of the 5,186 test rows
        for SpCas9 alone - and stays None, because a 0 there would read as a
        measured zero efficiency.
        """
        heads = tuple(self.predictions_checksum["heads"])
        rows: dict[int, tuple[str, dict[str, float | None]]] = {}
        for split in SPLITS:
            path = self.root / f"published_split/{split}.csv"
            with open(path, newline="", encoding="utf-8") as handle:
                for record in csv.DictReader(handle):
                    expect(
                        record["split"],
                        split,
                        what=f"{path.name}의 split 열",
                        source=path,
                    )
                    rows[int(record["row_index"])] = (
                        split,
                        {
                            head: (
                                None
                                if record[head].strip() == ""
                                else percent_to_fraction(float(record[head]))
                            )
                            for head in heads
                        },
                    )
        return rows

    @cached_property
    def _predictions(self) -> dict[int, dict[str, float]]:
        path = self.root / "v3_testset_predictions/predictions.csv"
        heads = tuple(self.predictions_checksum["heads"])
        predictions: dict[int, dict[str, float]] = {}
        with open(path, newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                predictions[int(record["row_index"])] = {
                    head: float(record[f"pred_{head}"]) for head in heads
                }
        expect(
            len(predictions),
            self.predictions_checksum["rows"],
            what="예측 행 수",
            source=path,
        )
        return predictions

    def split_role(self, row_id: int) -> str | None:
        """`train` / `val` / `test`, or None for a row the split files do not hold.

        Measured: the ISM 3,851 rows and the IG 400 rows are all `test` and
        nothing is unmatched. The display is kept as a cheap safeguard, not as
        a warning that is expected to fire.
        """
        found = self._split.get(int(row_id))
        return None if found is None else found[0]

    def label(self, row_id: int, head: str) -> float | None:
        """The measured efficiency as a fraction, or None where unmeasured.

        The split files store percent; None means that head has no measurement
        on that row, which a view states rather than filling in.
        """
        self.prediction_heads.index(head)
        found = self._split.get(int(row_id))
        if found is None:
            raise RowNotPresent(f"split 파일에 row_index {row_id}이(가) 없다")
        return found[1][head]

    def prediction(self, row_id: int, head: str) -> float:
        self.prediction_heads.index(head)
        try:
            return self._predictions[int(row_id)][head]
        except KeyError:
            raise RowNotPresent(
                f"SpCas9 예측 {len(self._predictions)}행에 row_index {row_id}이(가) 없다"
            ) from None

    # -- per-row results ---------------------------------------------------- #

    def wt_prediction(self, row_id: int, head: str) -> float:
        """ISM's own wild-type prediction, the check that catches a mis-join."""
        return float(
            self._ism["wt_prediction"][
                self.ism_rows.position(row_id), self.ism_heads.index(head)
            ]
        )

    def ism_profile(self, row_id: int, head: str) -> np.ndarray:
        """Per-position mean |delta| for one row and head, WT slot excluded."""
        position = self.ism_rows.position(row_id)
        head_index = self.ism_heads.index(head)
        return ism_position_profile(
            self._ism["delta"][position, :, :, head_index],
            self._ism["reference_base"][position],
        )

    def ism_profiles(self, row_ids: Sequence[int], head: str) -> np.ndarray:
        positions = self.ism_rows.positions(row_ids)
        head_index = self.ism_heads.index(head)
        return ism_position_profile(
            self._ism["delta"][positions][:, :, :, head_index],
            self._ism["reference_base"][positions],
        )

    def reference_sequence(self, row_id: int) -> str:
        bases = self._ism["reference_base"][self.ism_rows.position(row_id)]
        return "".join(BASES[int(base)] for base in bases)

    def has_ig(self, row_id: int) -> bool:
        return int(row_id) in self.ig_rows

    def ig_absence(self, row_id: int) -> IgAbsence | None:
        """The stated reason a row has no IG, for `explain` to print as such."""
        if self.has_ig(row_id):
            return None
        return IgAbsence(
            row_id=int(row_id),
            rows=self.ig_rows.rows,
            reason=f"이 행은 IG 고정 명단 {self.ig_rows.rows}행에 없음",
        )

    def ig_tokens(self, row_id: int, head: str) -> np.ndarray:
        """Token attribution for one row and head, on the standardized output."""
        return self._ig["attribution"][
            self.ig_rows.position(row_id), self.ig_heads.index(head)
        ].astype(np.float64)

    def ig_completeness_residual(self, row_id: int, head: str) -> float:
        return float(
            self._ig["completeness_residual"][
                self.ig_rows.position(row_id), self.ig_heads.index(head)
            ]
        )

    def rollout(self, row_id: int) -> np.ndarray:
        """One 5x5 token matrix. There is no head axis here (ADR 0008)."""
        if self._rollout["rollout"].ndim != 3:
            raise ArtifactMismatch("rollout에 head 축이 생겼다 (ADR 0008 위반)")
        return self._rollout["rollout"][self.rollout_rows.position(row_id)].astype(
            np.float64
        )

    def ig_stratum(self, row_id: int) -> str | None:
        """The stratum the frozen IG manifest assigns the row, from that manifest."""
        return self._ig_strata.get(int(row_id))

    @cached_property
    def _ig_strata(self) -> dict[int, str]:
        return {
            int(entry["row_index"]): str(entry["stratum"])
            for entry in self.ig_subset["entries"]
        }

    def ig_subset_rows(self, stratum: str) -> list[int]:
        self.require_stratum(stratum)
        return [row_id for row_id, name in self._ig_strata.items() if name == stratum]

    # -- recorded validation figures ---------------------------------------- #

    def validation_figures(
        self, stratum: str, head: str, variant: str = "magnitude"
    ) -> dict[str, Any]:
        """The recorded IG-ISM agreement for one stratum and head, with its denominator.

        One stratum at a time: `ig_ism_validation.json` declares
        `"pooling": "forbidden"`, so the generalisation stratum (200 rows, 193
        targets) and the PAM library stratum (200 rows, 5 targets) are never
        added together, and no figure here spans both.
        """
        self.ig_heads.index(head)
        declared = self.validation["strata"][self.require_stratum(stratum)]
        entry = declared["by_head"][head][variant]
        return {
            "stratum": stratum,
            "head": head,
            "variant": variant,
            "statistic": self.validation["statistic"],
            "rows": int(declared["rows"]),
            "distinct_protospacers": int(declared["distinct_protospacers"]),
            "per_row": entry["per_row"],
            "mean_profile": entry["mean_profile"],
            "agreement_sign": entry["agreement_sign"],
            "ism_token_profile": list(entry["mean_token_profile"]["ism"]),
            "ig_token_profile": list(entry["mean_token_profile"]["ig"]),
            "mismatch_note": self.validation["mismatch_note"],
        }

    def noise_floor_by_head(self, head: str) -> dict[str, Any]:
        """Replicate noise floor for one head. `single_constant_note` forbids pooling."""
        self.ism_heads.index(head)
        return dict(self.noise_floor["by_head"][head])

    @cached_property
    def claim_limit(self) -> str:
        """The claim limit the artifacts state, verbatim and identical across them."""
        stated = {
            str(self.cases_annotated["claim_limit"]),
            str(self.validation["claim_limit"]),
            str(self.head_order_reversal["claim_limit"]),
        }
        if len(stated) != 1:
            raise ArtifactMismatch(
                "SpCas9 artifact들의 claim_limit이 서로 다르다: 어느 것도 대표로 쓸 수 없다"
            )
        return stated.pop()


# --------------------------------------------------------------------------- #
# NT-Cas12a
# --------------------------------------------------------------------------- #


class Cas12aReader:
    """Reads the prepared NT-Cas12a Test1500 artifacts."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR) -> None:
        self.root = Path(data_dir) / "cas12a"
        if not self.root.is_dir():
            raise FileNotFoundError(f"Cas12a 자료가 없다: {self.root}")

    # -- light JSON --------------------------------------------------------- #

    @cached_property
    def metrics(self) -> dict[str, Any]:
        return load_json(self.root / "metrics.json")

    @cached_property
    def position_profiles(self) -> dict[str, Any]:
        return load_json(self.root / "position_profiles.json")

    @cached_property
    def token_profiles(self) -> dict[str, Any]:
        return load_json(self.root / "token_profiles.json")

    @cached_property
    def rollout_summary(self) -> dict[str, Any]:
        return load_json(self.root / "rollout_summary.json")

    @cached_property
    def agreement(self) -> dict[str, Any]:
        return load_json(self.root / "agreement.sliced.json")

    @cached_property
    def summary(self) -> dict[str, Any]:
        return load_json(self.root / "summary.json")

    @cached_property
    def cases_annotated(self) -> dict[str, Any]:
        return load_json(self.root / "cases_annotated.json")

    @property
    def strata(self) -> tuple[str, ...]:
        return CAS12A_STRATA

    @property
    def head(self) -> str:
        return CAS12A_HEAD

    # -- predictions and cases ---------------------------------------------- #

    @cached_property
    def _predictions(self) -> dict[int, dict[str, Any]]:
        path = self.root / "predictions.csv"
        rows: dict[int, dict[str, Any]] = {}
        with open(path, newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                expect(
                    record["output_unit"], "fraction", what="output_unit", source=path
                )
                rows[int(record["source_row"])] = {
                    "sequence": record["sequence"],
                    "label": float(record["label"]),
                    "raw_prediction": float(record["raw_prediction"]),
                    "clipped_prediction": float(record["clipped_prediction"]),
                    "pam_stratum": record["pam_stratum"],
                }
        return rows

    @cached_property
    def prediction_rows(self) -> RowAxis:
        return RowAxis(
            np.array(sorted(self._predictions), dtype=np.int64),
            "source_row",
            "Cas12a Test 전수",
        )

    def prediction(self, row_id: int) -> dict[str, Any]:
        """Label and prediction for one row, both fractions (`output_unit`)."""
        self.prediction_rows.position(row_id)
        return dict(self._predictions[int(row_id)])

    @cached_property
    def cases(self) -> list[dict[str, Any]]:
        """The 15 described cases, identified as the source data does: row + role."""
        path = self.root / "cases.csv"
        with open(path, newline="", encoding="utf-8") as handle:
            return [
                {
                    "source_row": int(record["source_row"]),
                    "role": record["role"],
                    "pam_stratum": record["pam_stratum"],
                    "label": float(record["label"]),
                    "raw_prediction": float(record["raw_prediction"]),
                    "raw_absolute_error": float(record["raw_absolute_error"]),
                    "rank_in_category": int(record["rank_in_category"]),
                    "selection_reason": record["selection_reason"],
                }
                for record in csv.DictReader(handle)
            ]

    # -- arrays ------------------------------------------------------------- #

    @cached_property
    def _ism(self) -> dict[str, np.ndarray]:
        keys = (
            "source_row",
            "accepted",
            "wt_prediction",
            "reference_base",
            "non_wt_mask",
            "base_valid_mask",
            "raw_delta",
            "denominator",
        )
        with np.load(self.root / "ism.npz") as archive:
            return {key: archive[key] for key in keys}

    @cached_property
    def _ig(self) -> dict[str, np.ndarray]:
        with np.load(self.root / "ig.npz") as archive:
            return {
                key: archive[key] for key in ("source_row", "accepted", "signed_mass")
            }

    @cached_property
    def _rollout(self) -> dict[str, np.ndarray]:
        with np.load(self.root / "rollout.npz") as archive:
            return {
                key: archive[key]
                for key in ("source_row", "accepted", "full", "sequence")
            }

    @cached_property
    def ism_rows(self) -> RowAxis:
        return RowAxis(self._ism["source_row"], "source_row", "Cas12a ISM 모집단")

    @cached_property
    def ig_rows(self) -> RowAxis:
        return RowAxis(self._ig["source_row"], "source_row", "Cas12a IG 모집단")

    @cached_property
    def rollout_rows(self) -> RowAxis:
        return RowAxis(
            self._rollout["source_row"], "source_row", "Cas12a rollout 모집단"
        )

    def ism_profile(self, row_id: int) -> np.ndarray:
        """Per-position mean |raw_delta| over the valid alternative bases.

        The WT slot is excluded structurally, from `reference_base`, and the
        divisor that excluding it produces is checked against the artifact's
        own stored `denominator`. So the exclusion is this reader's own code
        and the artifact still gets to contradict it.
        """
        position = self.ism_rows.position(row_id)
        alternatives = (
            self._ism["non_wt_mask"][position] & self._ism["base_valid_mask"][position]
        )
        stored = self._ism["denominator"][position]
        if not np.array_equal(alternatives.sum(axis=-1), stored):
            raise ArtifactMismatch(
                f"source_row {row_id}: 대체 염기 수가 저장된 denominator와 다르다"
            )
        return ism_position_profile(
            self._ism["raw_delta"][position],
            self._ism["reference_base"][position],
            valid=self._ism["base_valid_mask"][position],
        )

    def ig_signed_mass(self, row_id: int) -> np.ndarray:
        return self._ig["signed_mass"][self.ig_rows.position(row_id)].astype(np.float64)

    def rollout(self, row_id: int) -> np.ndarray:
        """The DNA-block matrix (`sequence`), CLS kept during propagation then dropped."""
        return self._rollout["sequence"][self.rollout_rows.position(row_id)].astype(
            np.float64
        )

    def accepted(self, row_id: int) -> bool:
        return bool(self._ism["accepted"][self.ism_rows.position(row_id)])

    # -- recorded figures --------------------------------------------------- #

    def require_stratum(self, stratum: str) -> str:
        """A PAM population has to be one of the three the artifacts hold."""
        if stratum not in self.strata:
            raise KeyError(
                f"Cas12a 집단 {stratum!r}이 없다 (있는 집단: {', '.join(self.strata)})"
            )
        return stratum

    def pools_pam_groups(self, stratum: str) -> bool:
        """Whether this population mixes the TTTV and TTTT groups.

        `all` does, and that is why its Spearman cannot be read as ranking
        ability: a view renders it with the pooling stated (`render.Stratum`).
        """
        return self.require_stratum(stratum) == "all"

    def metrics_for(self, stratum: str, clipping: str = "raw") -> dict[str, Any]:
        """Performance for one PAM population. `count` is the denominator used."""
        self.require_stratum(stratum)
        if clipping not in ("raw", "clipped"):
            raise KeyError(f"clipping은 raw 또는 clipped다: {clipping!r}")
        return dict(self.metrics[stratum][clipping])

    def position_profile(self, stratum: str) -> dict[str, Any]:
        entry = self.position_profiles[self.require_stratum(stratum)]
        return {
            "mean_abs_raw_delta": list(entry["mean_abs_raw_delta"]),
            "n_available": int(entry["n_available"]),
            "requested": int(entry["requested"]),
        }

    def rollout_degeneracy(self, stratum: str) -> dict[str, Any]:
        """The near-identical-query count, read as stored: a limit, not a failure."""
        entry = self.rollout_summary[self.require_stratum(stratum)]
        return {
            "near_identical_query_count": int(entry["near_identical_query_count"]),
            "n_accepted": int(entry["n_accepted"]),
        }

    @cached_property
    def claim_limit(self) -> str:
        """Cas12a states its limits in prose, not in a `claim_limit` key.

        The sections are quoted verbatim from `report.ko.md`; nothing is
        summarised or softened on the way to the screen.
        """
        text = (self.root / "report.ko.md").read_text(encoding="utf-8")
        sections = _markdown_sections(text)
        missing = [name for name in CAS12A_LIMIT_SECTIONS if name not in sections]
        if missing:
            raise ArtifactMismatch(f"report.ko.md에 {missing} 절이 없다")
        return "\n\n".join(
            f"[{name}]\n{sections[name].strip()}" for name in CAS12A_LIMIT_SECTIONS
        )


def _markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    heading: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections[heading] = "\n".join(body)
            heading = line[3:].strip()
            body = []
        elif heading is not None:
            body.append(line)
    if heading is not None:
        sections[heading] = "\n".join(body)
    return sections
