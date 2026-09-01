import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

# sklearn ships no py.typed marker, so pyright sees partially-unknown types here.
from sklearn.metrics import roc_auc_score  # pyright: ignore[reportUnknownVariableType]
from torch import nn

from knee.data import gold_studies
from knee.dicom import DicomDecodeError, load_volume
from knee.infer import merge_predictions
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.model import DEFAULT_BACKBONE, KneeModel, resolve_device
from knee.series import SeriesType, best_series_of_type
from knee.train_gold import TRAIN_SERIES_DIR, fit_head

# The three fluid specialists of the current ensemble (E001/E002).
DEFAULT_CV_SERIES_TYPES: tuple[SeriesType, ...] = (
    SeriesType.SAGITTAL_FLUID,
    SeriesType.CORONAL_FLUID,
    SeriesType.AXIAL_FLUID,
)


class GoldFeatureBank(BaseModel):
    """Cached per-plane study features for the gold set.

    Extract once, cross-validate many times: the backbone is frozen, so features are
    a pure function of the pixels and stay valid across fold counts, seeds, and
    repeats. One shared backbone serves every plane — equivalent to the per-plane
    models in `train_gold` because each of those starts from the same ImageNet
    weights and never trains its backbone.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    series_types: list[SeriesType]
    study_uids: list[str]
    # (n_studies, 12) float32 in LABEL_COLUMNS order, aligned with study_uids.
    labels: np.ndarray
    # Per plane, aligned with study_uids; None where the study lacks the plane or its
    # series failed to decode — the plane "sits out" for that study, as in inference.
    features: dict[SeriesType, list[torch.Tensor | None]]

    def plane_coverage(self) -> dict[SeriesType, int]:
        """Number of studies with usable features, per plane."""
        return {
            series_type: sum(feature is not None for feature in self.features[series_type])
            for series_type in self.series_types
        }


class GoldCVResult(BaseModel):
    """Pooled out-of-fold (OOF) cross-validation metrics on the gold studies."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_studies: int
    n_splits: int
    n_repeats: int
    seed: int
    series_types: list[SeriesType]
    plane_coverage: dict[SeriesType, int]
    # Mean across repeats; NaN where the gold labels carry a single class.
    per_label_auc: dict[str, float]
    # Mean across repeats of the per-repeat macro (labels with defined AUC only).
    macro_auc: float
    macro_auc_per_repeat: list[float]
    # (n_repeats, n_studies, 12) OOF probabilities, study axis aligned with the
    # bank's study order. Kept for analysis; deliberately carries no UIDs.
    oof_probabilities: np.ndarray


def collect_gold_features(
    comp_root: Path,
    *,
    series_types: Sequence[SeriesType] = DEFAULT_CV_SERIES_TYPES,
    model: KneeModel | None = None,
    backbone: str = DEFAULT_BACKBONE,
    input_size: int = 224,
    log: Callable[[str], None] = print,
) -> GoldFeatureBank:
    """Decode every gold study once per plane and cache pooled backbone features.

    Every gold study is kept, even when no plane decodes — at CV time such a study
    receives the same fallback constant the submission would, so the local metric
    measures exactly the pipeline that gets scored.

    Args:
        comp_root: Competition data root containing `train.csv`, `train_series.csv`,
            and `train_series/`.
        series_types: The ensemble's planes, one feature column each.
        model: Feature extractor; a fresh pretrained `KneeModel(backbone)` when
            omitted. Only its frozen backbone is used — the head is never touched.
        backbone: timm model name for the fresh model; ignored when `model` is given.
        input_size: Slice resize target fed to `load_volume`; must match the
            backbone's expectations (fixed-size ViTs reject other sizes).
        log: Progress sink (`print` in notebooks).

    Returns:
        The feature bank, aligned study-wise across labels and all planes.

    Raises:
        ValueError: If `series_types` is empty or contains duplicates (a duplicated
            plane would be double-weighted by the combiner at CV time).
    """
    if not series_types:
        raise ValueError("collect_gold_features needs at least one series type")
    if len(set(series_types)) != len(series_types):
        raise ValueError(f"Duplicate series types: {list(series_types)}")

    train_df = pd.read_csv(comp_root / "train.csv")
    series_df = pd.read_csv(comp_root / "train_series.csv")
    gold = gold_studies(train_df)
    log(f"{len(gold)} gold studies, planes {[t.value for t in series_types]}")

    model = model or KneeModel(backbone)
    model.freeze_backbone()
    device = resolve_device()
    model.to(device)
    model.eval()

    study_uids: list[str] = []
    label_rows: list[np.ndarray] = []
    features: dict[SeriesType, list[torch.Tensor | None]] = {t: [] for t in series_types}
    for position, (_, row) in enumerate(gold.iterrows(), start=1):
        study_uid = str(row[STUDY_ID_COLUMN])
        study_series = series_df[series_df[STUDY_ID_COLUMN] == study_uid]
        study_uids.append(study_uid)
        label_rows.append(row[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32))
        for series_type in series_types:
            series_uid = best_series_of_type(study_series, series_type)
            if series_uid is None:
                features[series_type].append(None)
                continue
            try:
                volume = load_volume(comp_root / TRAIN_SERIES_DIR / study_uid / series_uid, size=input_size)
                with torch.no_grad():  # backbone is frozen; cache features once
                    features[series_type].append(model.pool_features(volume.to(device)).cpu())
            except (ValueError, DicomDecodeError) as exc:
                features[series_type].append(None)
                log(f"{series_type}: skipping series of study {position}: {exc}")
        if position % 10 == 0:
            log(f"extracted {position}/{len(gold)}")

    return GoldFeatureBank(
        series_types=list(series_types),
        study_uids=study_uids,
        labels=np.stack(label_rows),
        features=features,
    )


def _stratified_fold_assignments(labels: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy multi-label stratification (simplified Sechidis et al. 2011).

    Labels are processed rarest-first — rare positives are the hardest to spread, so
    they pick folds before common ones consume the capacity. Each positive study goes
    to the fold currently holding the fewest positives of that label (ties broken by
    most remaining room), so every fold's training complement sees positives of every
    label wherever counts allow.

    Args:
        labels: (n_studies, n_labels) 0/1 array.
        n_splits: Number of folds; fold sizes differ by at most one.
        rng: Source of shuffling; fixes the assignment for a given seed.

    Returns:
        (n_studies,) int array of fold indices in [0, n_splits).
    """
    n_studies = labels.shape[0]
    assignment = np.full(n_studies, -1, dtype=np.int64)
    capacity = np.full(n_splits, n_studies // n_splits, dtype=np.int64)
    capacity[: n_studies % n_splits] += 1  # spread the remainder over the first folds
    fold_label_counts = np.zeros((n_splits, labels.shape[1]))

    label_order = [int(c) for c in np.argsort(labels.sum(axis=0)) if labels[:, int(c)].sum() > 0]
    for column in label_order:
        positives = np.flatnonzero((labels[:, column] > 0) & (assignment == -1))
        for study in rng.permutation(positives):
            open_folds = np.flatnonzero(capacity > 0)
            fold = min(open_folds, key=lambda f: (fold_label_counts[f, column], -capacity[f]))
            assignment[study] = fold
            capacity[fold] -= 1
            fold_label_counts[fold] += labels[study]
    for study in rng.permutation(np.flatnonzero(assignment == -1)):
        open_folds = np.flatnonzero(capacity > 0)
        fold = max(open_folds, key=lambda f: capacity[f])
        assignment[study] = fold
        capacity[fold] -= 1
    return assignment


def _oof_predictions(bank: GoldFeatureBank, assignment: np.ndarray, head_seed: int) -> np.ndarray:
    """One repeat's pooled OOF matrix: per fold, fit fresh heads and predict held-out.

    Mirrors the production pipeline exactly: per-plane linear heads trained with
    `fit_head` (as in `train_gold`), merged with `merge_predictions` (as in
    `predict_studies`) — including plane sit-outs and the 0.5 fallback for studies
    no plane could read.
    """
    n_studies = bank.labels.shape[0]
    targets = torch.from_numpy(bank.labels)  # pyright: ignore[reportUnknownMemberType]
    oof = np.full((n_studies, len(LABEL_COLUMNS)), np.nan)
    for fold in range(int(assignment.max()) + 1):
        train_indices = np.flatnonzero(assignment != fold)
        heads: dict[SeriesType, nn.Linear] = {}
        for plane_index, series_type in enumerate(bank.series_types):
            plane_features = bank.features[series_type]
            rows = [int(i) for i in train_indices if plane_features[int(i)] is not None]
            if not rows:
                continue  # plane absent from this training fold; it sits out
            stacked = torch.stack([f for i in rows if (f := plane_features[i]) is not None])
            torch.manual_seed(head_seed * 1000 + fold * 10 + plane_index)  # pyright: ignore[reportUnknownMemberType] # reproducible head init
            head = nn.Linear(stacked.shape[1], len(LABEL_COLUMNS))
            fit_head(head, stacked, targets[rows])
            heads[series_type] = head

        for study in np.flatnonzero(assignment == fold):
            per_model_rows: list[np.ndarray] = []
            for series_type in bank.series_types:
                feature = bank.features[series_type][int(study)]
                head = heads.get(series_type)
                if feature is None or head is None:
                    per_model_rows.append(np.full(len(LABEL_COLUMNS), np.nan))
                    continue
                with torch.no_grad():
                    per_model_rows.append(torch.sigmoid(head(feature)).numpy())
            oof[study] = merge_predictions(np.stack(per_model_rows), bank.series_types)
    return oof


def cross_validate_gold(
    bank: GoldFeatureBank,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> GoldCVResult:
    """Pooled-OOF stratified k-fold CV of the full ensemble on the gold studies.

    Per fold, fresh per-plane heads train on the fold's training studies and the
    production combiner merges their held-out predictions; each study's OOF row thus
    comes from models that never saw it. AUC is computed once over all pooled OOF
    rows — per-fold AUC at n≈12 held-out studies is undefined or hopelessly noisy
    for rare labels. Repeats re-run the whole procedure with reshuffled folds; the
    spread of the per-repeat macro is the error bar to read the mean against.

    Args:
        bank: Cached features from `collect_gold_features`.
        n_splits: Fold count; 5 keeps ≈46 training studies per fold at n=58.
        n_repeats: Independent repetitions with reshuffled folds.
        seed: Base seed; fold shuffling and head init derive from it, so results are
            reproducible for a given bank.
        log: Progress sink (`print` in notebooks).

    Returns:
        Aggregated CV metrics plus the raw OOF probabilities for deeper analysis.

    Raises:
        ValueError: If `n_splits` is not in [2, n_studies] or `n_repeats` < 1.
    """
    n_studies = bank.labels.shape[0]
    if not 2 <= n_splits <= n_studies:
        raise ValueError(f"n_splits must be in [2, {n_studies}], got {n_splits}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    # A label with a single class in the gold set has no defined AUC anywhere.
    defined = [c for c in range(len(LABEL_COLUMNS)) if len(np.unique(bank.labels[:, c])) == 2]

    oof_stack: list[np.ndarray] = []
    per_repeat_label_auc = np.full((n_repeats, len(LABEL_COLUMNS)), np.nan)
    macro_per_repeat: list[float] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        assignment = _stratified_fold_assignments(bank.labels, n_splits, rng)
        oof = _oof_predictions(bank, assignment, head_seed=seed + repeat)
        oof_stack.append(oof)
        for column in defined:
            score = roc_auc_score(bank.labels[:, column], oof[:, column])  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            per_repeat_label_auc[repeat, column] = float(score)  # pyright: ignore[reportUnknownArgumentType]
        macro = float(per_repeat_label_auc[repeat, defined].mean())
        macro_per_repeat.append(macro)
        log(f"repeat {repeat + 1}/{n_repeats}: macro OOF AUC {macro:.3f}")

    per_label_auc = {
        label: float(per_repeat_label_auc[:, column].mean()) if column in defined else math.nan
        for column, label in enumerate(LABEL_COLUMNS)
    }
    return GoldCVResult(
        n_studies=n_studies,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        series_types=bank.series_types,
        plane_coverage=bank.plane_coverage(),
        per_label_auc=per_label_auc,
        macro_auc=float(np.mean(macro_per_repeat)),
        macro_auc_per_repeat=macro_per_repeat,
        oof_probabilities=np.stack(oof_stack),
    )
