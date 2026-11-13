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
import re
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
CAS12A_AGREEMENT_SECTION = "설명과 IG–ISM 일치도"
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


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
class MethodPresence:
    """Whether one method covers a row, and - when it does not - why.

    `explain` prints an absence as an absence with its reason, so the reasons
    these artifacts actually record stay distinguishable: a row outside a
    frozen population (SpCas9 IG's 400-row list) and a method that ran and
    failed (Cas12a records `status: failed` with its own `failure` text).
    Rendered as one blank, both would read as the demo having lost a number.

    `population` names what decides the answer, with its size, because "no IG"
    means a different thing for each.

    `limit` is the third case, and it is not an absence: the method ran and its
    values are stored, but they cannot be read the way this view reads the
    others. Attention rollout is the real instance - it covers the row, and has
    no head axis at all (ADR 0008), so it can give no per-head reading. Calling
    that "없음" would understate real coverage; saying only "있음" would invite
    the per-head reading the artifact cannot support.
    """

    method: str
    present: bool
    population: str
    reason: str | None = None
    limit: str | None = None

    def __post_init__(self) -> None:
        if not self.present and not (self.reason or "").strip():
            raise ArtifactMismatch(
                f"{self.method}이(가) 없다고만 적혀 있다: 사유를 함께 적어야 한다"
            )

    def text(self) -> str:
        """One line for a view: present or not, with the population or the reason."""
        if not self.present:
            return f"{self.method}: 없음 — {self.reason}"
        line = f"{self.method}: 있음 — {self.population}"
        return f"{line} / 다만 {self.limit}" if self.limit else line


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
        path = self.root / "v3_report/head_order_reversal.json"
        report = load_json(path)
        expect(report["pooling"], "forbidden", what="pooling", source=path)
        # The artifact holds counts and distributions and no row list: a view
        # may quote it as a result, never mine it for cases to show.
        expect(
            report["selection_pool"], "forbidden", what="selection_pool", source=path
        )
        return report

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

    # -- cases, by the case table's own identifiers -------------------------- #

    def case_ids(self) -> tuple[str, ...]:
        """The case IDs the case table holds, in its order. No ID is minted here."""
        return tuple(str(case["case_id"]) for case in self.cases)

    @cached_property
    def case_strata(self) -> dict[str, dict[str, Any]]:
        """Each stratum's own selection pool, as the case table records it.

        The pool is per stratum and never summed: the two strata's quartile and
        decile bounds are their own (the gap scales differ about twofold), so a
        case's denominator is the stratum it was drawn from.
        """
        table = load_json(self.root / "v3_cases/cases.json")
        return {
            name: dict(entry) for name, entry in (table.get("strata") or {}).items()
        }

    def case(self, identifier: str) -> dict[str, Any]:
        """One selection record, by `case_id` or by its `row_index`.

        A row number resolves too, because `row_index` is the axis the
        artifacts share and a speaker reading one off another view should not
        have to translate it. Nothing else is accepted: the demo has no
        identifier of its own to look up.
        """
        wanted = str(identifier).strip()
        for case in self.cases:
            if wanted == str(case["case_id"]) or wanted == str(case["row_index"]):
                return dict(case)
        raise KeyError(
            f"SpCas9 사례 {identifier!r}이(가) 없다 "
            f"(있는 사례: {', '.join(self.case_ids())})"
        )

    def case_annotation(self, identifier: str) -> dict[str, Any]:
        """The attribution annotation for a case, addressed by `case_id`.

        `cases_annotated.json` points back at `cases.json` by `case_id` and
        holds no selection score, so this is the only direction the join runs:
        the selection is looked up first and its ID used here.
        """
        case_id = str(self.case(identifier)["case_id"])
        for annotation in self.cases_annotated["cases"]:
            if str(annotation["case_id"]) == case_id:
                return dict(annotation)
        raise ArtifactMismatch(f"주석 표에 사례 {case_id}이(가) 없다")

    @staticmethod
    def _in_population(
        method: str, row_id: int, axis_name: str, rows: int, present: bool
    ):
        """One method's presence, with the absence worded from its own population."""
        return MethodPresence(
            method=method,
            present=present,
            population=f"{axis_name} {rows:,}행",
            reason=None if present else f"이 행은 {axis_name} {rows:,}행에 없음",
        )

    def case_methods(self, identifier: str) -> tuple[MethodPresence, ...]:
        """Which methods cover this case's row, each absence with its reason."""
        row_id = int(self.case(identifier)["row_index"])
        annotation = self.case_annotation(identifier)

        # The annotation and the IG row axis are two records of the same fact.
        # If they disagree, neither reason on screen would be true, so this is
        # a mismatch to raise rather than a sentence to print.
        annotated = annotation.get("ig") is not None
        on_axis = self.has_ig(row_id)
        if annotated != on_axis:
            raise ArtifactMismatch(
                f"row_index {row_id}: 사례 주석의 IG 유무({annotated})와 "
                f"IG 고정 명단 수록 여부({on_axis})가 다르다"
            )

        return (
            self._in_population(
                "예측",
                row_id,
                "예측 표",
                len(self._predictions),
                row_id in self._predictions,
            ),
            self._in_population(
                "ISM", row_id, "ISM sweep", self.ism_rows.rows, row_id in self.ism_rows
            ),
            MethodPresence(
                method="IG",
                present=on_axis,
                population=f"IG 고정 명단 {self.ig_rows.rows:,}행",
                reason=None if on_axis else self.ig_absence(row_id).reason,
            ),
            MethodPresence(
                method="rollout",
                present=row_id in self.rollout_rows,
                population=f"rollout 모집단 {self.rollout_rows.rows:,}행",
                reason=(
                    None
                    if row_id in self.rollout_rows
                    else f"이 행은 rollout 모집단 {self.rollout_rows.rows:,}행에 없음"
                ),
                limit=(
                    "head 축이 없어 head별 읽기를 줄 수 없다 (ADR 0008). 그래서 "
                    "사례 주석의 입력이 아니고, 이 뷰에 head별 값이 없다."
                ),
            ),
        )

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

    @property
    def split_rows(self) -> int:
        """How many rows the three split files hold together: the role's denominator."""
        return len(self._split)

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
            reason=(
                f"이 행은 IG 고정 명단 {self.ig_rows.rows:,}행에 없음 "
                "(명단은 사례 선정 전에 자기 seed로 고정됐고, 사례를 명단에 맞춰 "
                "고르거나 버리지 않았다)"
            ),
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
        listed = len(self.ig_subset_rows(stratum))
        if int(declared["rows"]) != listed:
            raise ArtifactMismatch(
                f"{stratum} 층: ig_ism_validation이 적은 행 수({declared['rows']})와 "
                f"IG 고정 명단의 행 수({listed})가 다르다"
            )
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

    def stratum_ism_profile(self, stratum: str, head: str) -> dict[str, Any]:
        """One stratum's mean ISM position profile for one head, WT slot excluded.

        The validation report records this profile only at token resolution,
        so the per-position one is built here from the stratum's IG-subset rows
        - the same rows the recorded agreement is over, so the denominator is
        the stratum's and never the 3,851-row sweep. Built, it is checked
        against the recorded token profile before anything is shown: a profile
        whose token means are not the analysis' own is a misread (a wrong head,
        a wrong row, the WT slot averaged in), and is refused rather than drawn.
        """
        recorded = self.validation_figures(stratum, head)
        profile = self.ism_profiles(self.ig_subset_rows(stratum), head).mean(axis=0)
        tokens = aggregate_to_tokens(profile, self.token_boundaries)
        if not np.allclose(tokens, recorded["ism_token_profile"], rtol=1e-5, atol=0):
            raise ArtifactMismatch(
                f"{stratum} 층 {head}: 위치 프로파일의 토큰 평균 {tokens.tolist()}이 "
                f"기록된 ISM 토큰 프로파일 {recorded['ism_token_profile']}과 다르다"
            )
        return {
            "stratum": stratum,
            "head": head,
            "profile": [float(value) for value in profile],
            "rows": recorded["rows"],
            "distinct_protospacers": recorded["distinct_protospacers"],
        }

    def noise_floor_by_head(self, head: str) -> dict[str, Any]:
        """Replicate noise floor for one head. `single_constant_note` forbids pooling."""
        self.ism_heads.index(head)
        return dict(self.noise_floor["by_head"][head])

    def reversal_stratum(self, stratum: str) -> dict[str, Any]:
        """The recorded head-ordering reversal for one stratum. Never a combined entry.

        `head_order_reversal.json` declares `"pooling": "forbidden"` like the
        validation report does: the PAM library stratum's rows are variants of
        5 targets, so its row count is never added to the other stratum's.
        """
        declared = self.head_order_reversal["strata"]
        if self.require_stratum(stratum) not in declared:
            raise ArtifactMismatch(
                f"head 순서 역전에 층 {stratum!r}의 항목이 없다 "
                f"(있는 항목: {', '.join(declared)})"
            )
        return dict(declared[stratum])

    def reversal_floor(self) -> dict[str, Any]:
        """The noise floor the reversal applied, checked against the floor artifact.

        The reversal quotes the floor it tested label gaps against. It has to
        be the per-head median of `v3_noise_floor.json` - `single_constant_note`
        forbids the pooled constant, which is several times evoCas9's own - and
        the gap it tests has to be the measured one. A quoted floor that is not
        the artifact's per-head value is refused, not shown.
        """
        source = self.root / "v3_report/head_order_reversal.json"
        floor = self.head_order_reversal["floor"]
        expect(floor["statistic"], "median", what="floor.statistic", source=source)
        expect(
            floor["pooled_applied"], False, what="floor.pooled_applied", source=source
        )
        expect(
            floor["gap_source"],
            "measured_labels",
            what="floor.gap_source",
            source=source,
        )
        expect(
            self.noise_floor["label_scale"],
            "fraction",
            what="label_scale",
            source=self.root / "v3_noise_floor.json",
        )
        for head in self.analysed_heads:
            quoted = float(floor["by_head"][head])
            own = float(self.noise_floor_by_head(head)["median"])
            if quoted != own:
                raise ArtifactMismatch(
                    f"head 순서 역전이 적용한 {head} 노이즈 바닥({quoted})이 "
                    f"v3_noise_floor.json의 head별 median({own})과 다르다"
                )
        return dict(floor)

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

    @cached_property
    def _layout(self) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
        """DNA token spans and PAM span from `layout.json`, checked on load.

        `prepare` lifted these from the IG shards (pinned equal across all of
        them), so the screen can draw each IG token at its recorded bases
        instead of re-deriving a 6-mer split. The spans are refused, never
        re-sorted, when out of position order: their order is IG's token
        column order, and sorting would silently re-pair values and bases.
        """
        path = self.root / "layout.json"
        payload = load_json(path)
        half_open = "0-based half-open"
        expect(payload.get("coordinates"), half_open, what="coordinates", source=path)
        layout = payload.get("layout", {})
        expect(
            layout.get("coordinate_system"),
            half_open,
            what="layout.coordinate_system",
            source=path,
        )
        spans = tuple((int(start), int(end)) for start, end in payload["dna_spans"])
        expect(
            tuple(start for start, _ in spans),
            (0, *(end for _, end in spans[:-1])),
            what="dna_spans의 시작 위치 (0부터 빈틈없이 위치 순)",
            source=path,
        )
        if not spans or any(end <= start for start, end in spans):
            raise ArtifactMismatch(f"{path}: dna_spans에 빈 구간이 있다: {spans}")
        start, end = (int(value) for value in layout["pam"])
        if not 0 <= start < end <= spans[-1][1]:
            raise ArtifactMismatch(
                f"{path}: pam {[start, end]}이(가) dna_spans 범위 "
                f"[0, {spans[-1][1]}) 밖이다"
            )
        return spans, (start, end)

    @property
    def token_spans(self) -> tuple[tuple[int, int], ...]:
        """The DNA tokens' base spans, 0-based half-open, in position order."""
        return self._layout[0]

    @property
    def pam_span(self) -> tuple[int, int]:
        """The recorded PAM span, 0-based half-open (on the real data, `[4, 8]`)."""
        return self._layout[1]

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

    # -- cases, identified as the source data identifies them ---------------- #

    def case_ids(self) -> tuple[str, ...]:
        """`<source_row>:<role>` for each described case, in the table's order.

        The source data identifies a case by its row and the role it was
        selected for, and a row can hold more than one role, so both are needed
        to address one. The demo joins them with a colon and mints nothing.
        """
        return tuple(f"{case['source_row']}:{case['role']}" for case in self.cases)

    def case(self, identifier: str) -> dict[str, Any]:
        """One case, by `<source_row>:<role>` or by the row alone where unique."""
        wanted = str(identifier).strip()
        matches = [
            case
            for case in self.cases
            if wanted == f"{case['source_row']}:{case['role']}"
            or wanted == str(case["source_row"])
        ]
        if not matches:
            raise KeyError(
                f"Cas12a 사례 {identifier!r}이(가) 없다 "
                f"(있는 사례: {', '.join(self.case_ids())})"
            )
        if len(matches) > 1:
            raise KeyError(
                f"source_row {identifier}은(는) 역할이 여럿이다: "
                + ", ".join(f"{case['source_row']}:{case['role']}" for case in matches)
            )
        return dict(matches[0])

    @cached_property
    def _annotations(self) -> dict[tuple[int, str], dict[str, Any]]:
        return {
            (int(case["source_row"]), str(case["role"])): case
            for case in self.cases_annotated["cases"]
        }

    def case_annotation(self, identifier: str) -> dict[str, Any]:
        """The method-by-method annotation for one case, keyed by row *and* role.

        Both parts of the key are needed: a row can be described under more
        than one role, and the annotation records the methods per described
        case rather than per row.
        """
        case = self.case(identifier)
        key = (int(case["source_row"]), str(case["role"]))
        if key not in self._annotations:
            raise ArtifactMismatch(f"주석 표에 사례 {key[0]}:{key[1]}이(가) 없다")
        return dict(self._annotations[key])

    @cached_property
    def case_categories(self) -> dict[str, dict[str, Any]]:
        """Each role's pool and how many of it were selected.

        A case list without its pool invites reading 5 cases as 5 findings:
        5 of 16 out-of-range rows and 5 of 375 high-error rows are different
        claims, and the pool sizes are the artifact's own.
        """
        return dict(load_json(self.root / "cases.json").get("categories") or {})

    def case_pool(self, role: str) -> dict[str, int]:
        """The count a role's cases were actually drawn from, and what bounded it.

        `available` is the denominator, not `eligible`: the protocol sets
        `exclude_previous`, so a later category picks from the rows an earlier
        one did not already take - low_error's 375 eligible rows are 372 by the
        time it draws. `eligible` is kept beside it, because the gap between
        the two *is* the exclusion and hiding it would make the pool look
        untouched. A role with no recorded pool is refused: inventing one, or
        quietly falling back to the Test population, would put a denominator on
        screen that nothing measured.
        """
        category = self.case_categories.get(role)
        if not category:
            raise ArtifactMismatch(
                f"cases.json에 역할 {role!r}의 후보 모집단 기록이 없다"
            )
        if "available" not in category:
            raise ArtifactMismatch(
                f"역할 {role!r}의 기록에 available이 없다: 실제 분모를 적을 수 없다"
            )
        return {
            "available": int(category["available"]),
            "eligible": int(category.get("eligible", category["available"])),
            "selected": int(category.get("selected", 0)),
        }

    def case_methods(self, identifier: str) -> tuple[MethodPresence, ...]:
        """Each method's recorded `status` for this case, failures quoted.

        The statuses are the annotation's, not re-derived: a method that ran
        and failed records its own `failure` text, and that text is what
        `explain` shows instead of a blank.
        """
        annotation = self.case_annotation(identifier)
        methods = annotation.get("methods") or {}
        population = f"Test 전수 {self.prediction_rows.rows:,}행"
        entries = [
            MethodPresence(
                method="예측",
                present=int(annotation["source_row"]) in self._predictions,
                population=population,
                reason=(
                    None
                    if int(annotation["source_row"]) in self._predictions
                    else f"이 행은 {population}에 없음"
                ),
            )
        ]
        for key, label in (("ism", "ISM"), ("ig", "IG"), ("rollout", "rollout")):
            entry = methods.get(key)
            if entry is None:
                entries.append(
                    MethodPresence(
                        method=label,
                        present=False,
                        population=population,
                        reason=f"주석 표에 {key} 항목이 없음",
                    )
                )
                continue
            status = str(entry.get("status"))
            failure = entry.get("failure")
            entries.append(
                MethodPresence(
                    method=label,
                    present=status == "available",
                    population=population,
                    reason=(
                        None
                        if status == "available"
                        else f"status={status}"
                        + (f": {failure}" if failure else " (사유 기록 없음)")
                    ),
                )
            )
        return tuple(entries)

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

    def agreement_primary(self, stratum: str) -> dict[str, Any]:
        """The protocol's primary IG-ISM statistic for one PAM population.

        `agreement.json` records eight variants per population (token set x
        support x mass/density); `protocol.primary` names the one the analysis
        reports, as `<tokens>/<support>/<kind>`, while `versions` nests them
        support -> tokens -> kind. The variant is looked up by those names and
        the population's own `primary` entry has to be that variant, so a
        reader that picked another variant, or the wrong nesting, fails here
        instead of showing a neighbouring number.
        """
        protocol = self.agreement["protocol"]
        primary = str(protocol["primary"])
        tokens, support, kind = primary.split("/")
        entry = self.agreement["populations"][self.require_stratum(stratum)]
        variant = entry["versions"][support][tokens][kind]
        sequences = variant["sequences"]
        for key in ("median", "n_valid", "n_total"):
            if entry["primary"][key] != sequences[key]:
                raise ArtifactMismatch(
                    f"agreement {stratum}: primary의 {key}({entry['primary'][key]})가 "
                    f"protocol이 지정한 {primary}의 값({sequences[key]})과 다르다"
                )
        correlation = variant["mean_profile"]["correlation"]
        return {
            "primary": primary,
            "median": sequences["median"],
            "n_valid": int(sequences["n_valid"]),
            "n_total": int(sequences["n_total"]),
            "reason": sequences.get("reason"),
            "mean_profile_rho": correlation.get("rho"),
            "mean_profile_tokens": int(correlation["n_valid"]),
            "mean_profile_reason": correlation.get("reason"),
            # The mean profile's own row count: the fewest rows any token was
            # paired on, not the per-row statistic's `n_valid`.
            "mean_profile_rows": min(
                int(token["n_paired"]) for token in variant["mean_profile"]["profile"]
            ),
        }

    @cached_property
    def agreement_section(self) -> dict[str, Any]:
        """The report's own IG-ISM section: its prose, and the figures it links.

        Cas12a records no `mismatch_note` key; the analysis states what a low
        agreement means in this section of `report.ko.md`, so that prose is
        what a view quotes beside the number, verbatim. The SVGs are the ones
        this section links, resolved under `data/cas12a/` - named by path,
        never opened.
        """
        sections = _markdown_sections(
            (self.root / "report.ko.md").read_text(encoding="utf-8")
        )
        if CAS12A_AGREEMENT_SECTION not in sections:
            raise ArtifactMismatch(
                f"report.ko.md에 {CAS12A_AGREEMENT_SECTION} 절이 없다"
            )
        prose: list[str] = []
        images: list[tuple[str, Path]] = []
        for line in sections[CAS12A_AGREEMENT_SECTION].splitlines():
            image = _MARKDOWN_IMAGE.fullmatch(line.strip())
            if image:
                images.append((image.group(1), self.root / image.group(2)))
            elif line.strip():
                prose.append(line.strip())
        if not prose:
            raise ArtifactMismatch(
                f"report.ko.md의 {CAS12A_AGREEMENT_SECTION} 절이 비어 있다"
            )
        return {"title": CAS12A_AGREEMENT_SECTION, "prose": prose, "images": images}

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
