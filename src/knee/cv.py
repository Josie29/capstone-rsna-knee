import math
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

# sklearn ships no py.typed marker, so pyright sees partially-unknown types here.
from sklearn.metrics import roc_auc_score  # pyright: ignore[reportUnknownVariableType]
from torch import nn

from knee.dicom import DicomDecodeError, load_volume
from knee.fitting import fit_attention_head, fit_head
from knee.infer import merge_predictions
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.model import (
    DEFAULT_BACKBONE,
    HeadType,
    KneeModel,
    PerLabelAttentionHead,
    resolve_device,
)
from knee.series import TRAIN_SERIES_DIR, SeriesType, best_series_of_type

# The three fluid specialists of the current ensemble (E001/E002/E003).
DEFAULT_CV_SERIES_TYPES: tuple[SeriesType, ...] = (
    SeriesType.SAGITTAL_FLUID,
    SeriesType.CORONAL_FLUID,
    SeriesType.AXIAL_FLUID,
)

# Soft labels are binarized at this threshold wherever a binary quantity is needed
# (stratification, AUC). Training always uses the un-thresholded probabilities.
EVAL_THRESHOLD = 0.5

# DICOM decode is CPU-bound and serial per slice; pylibjpeg's C decoders release the
# GIL, so threads genuinely parallelize it across the machine's cores.
_DECODE_WORKERS = 4


class FeatureBank(BaseModel):
    """Cached per-plane, per-slice study features for a labeled study set.

    Extract once, train/cross-validate many times: the backbone is frozen, so
    features are a pure function of the pixels and stay valid across fold counts,
    seeds, and repeats. One shared backbone serves every plane — equivalent to the
    per-plane models in training because each of those starts from the same ImageNet
    weights and never trains its backbone. Features are cached *per slice* (pooling
    deferred to fit time) so any pooler — mean+max or attention — trains from the
    same bank; `pooled_view` bridges to the mean+max path.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    series_types: list[SeriesType]
    study_uids: list[str]
    # (n_studies, 12) float32 in LABEL_COLUMNS order, aligned with study_uids.
    # Hard 0/1 for gold labels, probabilities in [0, 1] for blended labels.
    labels: np.ndarray
    # Per plane, aligned with study_uids: (n_slices_i, feature_dim) matrices, slice
    # counts varying per study; None where the study lacks the plane or its series
    # failed to decode — the plane "sits out" for that study, as in inference.
    features: dict[SeriesType, list[torch.Tensor | None]]

    def plane_coverage(self) -> dict[SeriesType, int]:
        """Number of studies with usable features, per plane."""
        return {
            series_type: sum(feature is not None for feature in self.features[series_type])
            for series_type in self.series_types
        }


class CVResult(BaseModel):
    """Pooled out-of-fold (OOF) cross-validation metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_studies: int
    n_splits: int
    n_repeats: int
    seed: int
    series_types: list[SeriesType]
    plane_coverage: dict[SeriesType, int]
    # Mean across repeats; NaN where the thresholded labels carry a single class.
    per_label_auc: dict[str, float]
    # Mean across repeats of the per-repeat macro (labels with defined AUC only).
    macro_auc: float
    macro_auc_per_repeat: list[float]
    # (n_repeats, n_studies, 12) OOF probabilities, study axis aligned with the
    # bank's study order. Kept for analysis; deliberately carries no UIDs.
    oof_probabilities: np.ndarray


def collect_features(
    comp_root: Path,
    labels: pd.DataFrame,
    *,
    series_types: Sequence[SeriesType] = DEFAULT_CV_SERIES_TYPES,
    model: KneeModel | None = None,
    backbone: str = DEFAULT_BACKBONE,
    input_size: int = 224,
    crop_mm: float | None = None,
    log: Callable[[str], None] = print,
) -> FeatureBank:
    """Decode every labeled study once per plane and cache pooled backbone features.

    Every study is kept, even when no plane decodes — at CV time such a study
    receives the same fallback constant the submission would, so the local metric
    measures exactly the pipeline that gets scored. Decodes run on a small thread
    pool with one-study lookahead (the `predict_studies` pattern) so CPU decode
    overlaps the backbone forwards.

    Args:
        comp_root: Competition data root containing `train_series.csv` and
            `train_series/`.
        labels: One row per study: `StudyInstanceUID` plus the 12 label columns as
            floats — `gold_studies` or `load_blended_labels` output.
        series_types: The ensemble's planes, one feature column each.
        model: Feature extractor; a fresh pretrained `KneeModel(backbone)` when
            omitted. Only its frozen backbone is used — the head is never touched.
        backbone: timm model name for the fresh model; ignored when `model` is given.
        input_size: Slice resize target fed to `load_volume`; must match the
            backbone's expectations (fixed-size ViTs reject other sizes).
        crop_mm: Fixed-mm crop fed to `load_volume` (None = full frame). Checkpoints
            trained from this bank must be saved with the same value.
        log: Progress sink (`print` in notebooks).

    Returns:
        The feature bank, aligned study-wise across labels and all planes.

    Raises:
        ValueError: If `series_types` is empty or contains duplicates (a duplicated
            plane would be double-weighted by the combiner at CV time).
    """
    if not series_types:
        raise ValueError("collect_features needs at least one series type")
    if len(set(series_types)) != len(series_types):
        raise ValueError(f"Duplicate series types: {list(series_types)}")

    series_df = pd.read_csv(comp_root / "train_series.csv")
    # One dict lookup per study instead of a full-frame boolean scan (24k rows x 4.4k
    # studies would dominate the loop's Python time).
    series_groups: dict[str, pd.DataFrame] = {
        str(uid): group for uid, group in series_df.groupby(STUDY_ID_COLUMN)
    }
    no_series = series_df.iloc[0:0]

    study_uids = [str(uid) for uid in labels[STUDY_ID_COLUMN]]
    label_matrix = labels[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    log(f"{len(study_uids)} studies, planes {[t.value for t in series_types]}")

    model = model or KneeModel(backbone)
    model.freeze_backbone()
    device = resolve_device()
    model.to(device)
    model.eval()

    features: dict[SeriesType, list[torch.Tensor | None]] = {t: [] for t in series_types}

    def decode(position: int, series_type: SeriesType) -> torch.Tensor | None:
        """The chosen series' volume, or None when the plane sits out for this study."""
        study_uid = study_uids[position]
        series_uid = best_series_of_type(series_groups.get(study_uid, no_series), series_type)
        if series_uid is None:
            return None
        try:
            return load_volume(
                comp_root / TRAIN_SERIES_DIR / study_uid / series_uid, size=input_size, crop_mm=crop_mm
            )
        except (ValueError, DicomDecodeError) as exc:
            log(f"{series_type}: skipping series of study {position + 1}: {exc}")
            return None

    with ThreadPoolExecutor(max_workers=_DECODE_WORKERS) as executor:

        def submit_decodes(position: int) -> list[Future[torch.Tensor | None]]:
            return [executor.submit(decode, position, t) for t in series_types]

        pending = submit_decodes(0) if study_uids else []
        for position in range(len(study_uids)):
            volumes = [future.result() for future in pending]
            if position + 1 < len(study_uids):
                # Queue the next study's decodes so they overlap the forwards below.
                pending = submit_decodes(position + 1)
            for series_type, volume in zip(series_types, volumes, strict=True):
                if volume is None:
                    features[series_type].append(None)
                    continue
                with torch.no_grad():  # backbone is frozen; cache features once
                    features[series_type].append(model.slice_features(volume.to(device)).cpu())
            if (position + 1) % 10 == 0:
                log(f"extracted {position + 1}/{len(study_uids)}")

    return FeatureBank(
        series_types=list(series_types),
        study_uids=study_uids,
        labels=label_matrix,
        features=features,
    )


def pooled_view(slice_matrix: torch.Tensor) -> torch.Tensor:
    """Mean+max pooling of one study's slice matrix — the E001-E004 study vector.

    Args:
        slice_matrix: (n_slices, feature_dim) cached slice features.

    Returns:
        1-D tensor of length `2 * feature_dim`, identical to what `pool_features`
        would have produced from the same volume.
    """
    return torch.cat([slice_matrix.mean(dim=0), slice_matrix.max(dim=0).values])


def save_feature_bank(bank: FeatureBank, path: Path) -> None:
    """Persist a feature bank so head training/CV can rerun without any decoding.

    Slice matrices are stored fp16 to halve the file (~0.5GB at 4.4k studies x 3
    planes); `load_feature_bank` casts back to fp32.

    Args:
        bank: The bank to save.
        path: Destination .pt file.
    """
    torch.save(
        {
            "series_types": [t.value for t in bank.series_types],
            "study_uids": bank.study_uids,
            "labels": torch.from_numpy(bank.labels),  # pyright: ignore[reportUnknownMemberType]
            "features": {
                t.value: [f.half() if f is not None else None for f in bank.features[t]]
                for t in bank.series_types
            },
        },
        path,
    )


def load_feature_bank(path: Path) -> FeatureBank:
    """Rebuild a feature bank saved by `save_feature_bank`.

    Args:
        path: The .pt file.

    Returns:
        The bank, with the same study alignment it was saved with.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    series_types = [SeriesType(value) for value in payload["series_types"]]
    return FeatureBank(
        series_types=series_types,
        study_uids=[str(uid) for uid in payload["study_uids"]],
        labels=payload["labels"].numpy(),
        features={
            t: [f.float() if f is not None else None for f in payload["features"][t.value]]
            for t in series_types
        },
    )


def _stratified_fold_assignments(labels: np.ndarray, n_splits: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy multi-label stratification (simplified Sechidis et al. 2011).

    Labels are processed rarest-first — rare positives are the hardest to spread, so
    they pick folds before common ones consume the capacity. Each positive study goes
    to the fold currently holding the fewest positives of that label (ties broken by
    most remaining room), so every fold's training complement sees positives of every
    label wherever counts allow.

    Args:
        labels: (n_studies, n_labels) 0/1 array — soft labels must be thresholded by
            the caller first.
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


def fit_plane_head(
    slice_features: list[torch.Tensor],
    targets: torch.Tensor,
    head_type: HeadType,
    seed: int,
) -> nn.Module:
    """Fit one plane's head of the requested type from cached slice matrices.

    Args:
        slice_features: Per-study (n_slices_i, feature_dim) matrices for this plane.
        targets: (n_studies, 12) float labels aligned with `slice_features`.
        head_type: Which pooling/head family to fit.
        seed: Head-init/shuffle seed for reproducibility.

    Returns:
        The trained head in eval mode.
    """
    feature_dim = slice_features[0].shape[1]
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType] # reproducible head init
    if head_type is HeadType.ATTENTION:
        attention = PerLabelAttentionHead(feature_dim)
        fit_attention_head(attention, slice_features, targets, seed=seed)
        return attention
    linear = nn.Linear(2 * feature_dim, len(LABEL_COLUMNS))
    fit_head(linear, torch.stack([pooled_view(m) for m in slice_features]), targets)
    return linear


def predict_with_head(head: nn.Module, slice_matrix: torch.Tensor) -> np.ndarray:
    """One study's 12 probabilities from a head fit by `fit_plane_head`."""
    with torch.no_grad():
        if isinstance(head, PerLabelAttentionHead):
            return torch.sigmoid(head(slice_matrix)).numpy()
        return torch.sigmoid(head(pooled_view(slice_matrix))).numpy()


def stratified_holdout(labels: np.ndarray, *, val_fraction: float = 0.1, seed: int = 0) -> np.ndarray:
    """One stratified validation mask, for regimes where full CV is infeasible.

    Reuses the multi-label fold assigner: splits into round(1/val_fraction) folds
    and takes fold 0 as validation, so rare labels land on both sides of the split
    with the same guarantee the CV protocol gives.

    Args:
        labels: (n_studies, 12) labels; soft values are thresholded at
            `EVAL_THRESHOLD` for stratification.
        val_fraction: Approximate validation share.
        seed: Assignment seed, so every arm of an experiment shares the split.

    Returns:
        (n_studies,) bool mask, True = validation.

    Raises:
        ValueError: If `val_fraction` is not in (0, 0.5].
    """
    if not 0 < val_fraction <= 0.5:
        raise ValueError(f"val_fraction must be in (0, 0.5], got {val_fraction}")
    binary = (labels >= EVAL_THRESHOLD).astype(np.float32)
    n_splits = round(1 / val_fraction)
    assignment = _stratified_fold_assignments(binary, n_splits, np.random.default_rng(seed))
    return assignment == 0


def _oof_predictions(
    bank: FeatureBank, assignment: np.ndarray, head_seed: int, head_type: HeadType
) -> np.ndarray:
    """One repeat's pooled OOF matrix: per fold, fit fresh heads and predict held-out.

    Mirrors the production pipeline exactly: per-plane heads of `head_type` trained
    on the bank's (possibly soft) labels, merged with `merge_predictions` (as in
    `predict_studies`) — including plane sit-outs and the 0.5 fallback for studies
    no plane could read.
    """
    n_studies = bank.labels.shape[0]
    targets = torch.from_numpy(bank.labels)  # pyright: ignore[reportUnknownMemberType]
    oof = np.full((n_studies, len(LABEL_COLUMNS)), np.nan)
    for fold in range(int(assignment.max()) + 1):
        train_indices = np.flatnonzero(assignment != fold)
        heads: dict[SeriesType, nn.Module] = {}
        for plane_index, series_type in enumerate(bank.series_types):
            plane_features = bank.features[series_type]
            rows = [int(i) for i in train_indices if plane_features[int(i)] is not None]
            if not rows:
                continue  # plane absent from this training fold; it sits out
            matrices = [f for i in rows if (f := plane_features[i]) is not None]
            heads[series_type] = fit_plane_head(
                matrices, targets[rows], head_type, seed=head_seed * 1000 + fold * 10 + plane_index
            )

        for study in np.flatnonzero(assignment == fold):
            per_model_rows: list[np.ndarray] = []
            for series_type in bank.series_types:
                feature = bank.features[series_type][int(study)]
                head = heads.get(series_type)
                if feature is None or head is None:
                    per_model_rows.append(np.full(len(LABEL_COLUMNS), np.nan))
                    continue
                per_model_rows.append(predict_with_head(head, feature))
            oof[study] = merge_predictions(np.stack(per_model_rows), bank.series_types)
    return oof


def cross_validate(
    bank: FeatureBank,
    *,
    head_type: HeadType = HeadType.MEAN_MAX,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> CVResult:
    """Pooled-OOF stratified k-fold CV of the full ensemble on the bank's studies.

    Per fold, fresh per-plane heads train on the fold's training studies and the
    production combiner merges their held-out predictions; each study's OOF row thus
    comes from models that never saw it. Heads fit on the bank's labels as-is (soft
    probabilities stay soft); stratification and AUC use the labels thresholded at
    `EVAL_THRESHOLD`, since both need a binary quantity. AUC is computed once over
    all pooled OOF rows; repeats re-run the whole procedure with reshuffled folds —
    the spread of the per-repeat macro is the error bar to read the mean against.

    When the bank's labels are miner-derived (blended), the resulting AUC measures
    agreement with the report miner, not ground truth — a consistent model-selection
    signal, with the leaderboard as the truth check.

    Args:
        bank: Cached features from `collect_features` (or `load_feature_bank`).
        head_type: Pooling/head family to fit per fold — `mean_max` reproduces the
            E001-E004 pipeline; `attention` fits per-label attention MIL heads. Same
            folds either way, so results A/B cleanly for a given seed.
        n_splits: Fold count.
        n_repeats: Independent repetitions with reshuffled folds. Repeats only set
            the error-bar precision (measured spread at n=4.4k is ~±0.001), so 3
            suffices; means stay comparable with earlier 5-repeat rows.
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

    binary = (bank.labels >= EVAL_THRESHOLD).astype(np.float32)
    # A label with a single thresholded class has no defined AUC anywhere.
    defined = [c for c in range(len(LABEL_COLUMNS)) if len(np.unique(binary[:, c])) == 2]

    oof_stack: list[np.ndarray] = []
    per_repeat_label_auc = np.full((n_repeats, len(LABEL_COLUMNS)), np.nan)
    macro_per_repeat: list[float] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        assignment = _stratified_fold_assignments(binary, n_splits, rng)
        oof = _oof_predictions(bank, assignment, head_seed=seed + repeat, head_type=head_type)
        oof_stack.append(oof)
        for column in defined:
            score = roc_auc_score(binary[:, column], oof[:, column])  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
            per_repeat_label_auc[repeat, column] = float(score)  # pyright: ignore[reportUnknownArgumentType]
        macro = float(per_repeat_label_auc[repeat, defined].mean())
        macro_per_repeat.append(macro)
        log(f"repeat {repeat + 1}/{n_repeats}: macro OOF AUC {macro:.3f}")

    per_label_auc = {
        label: float(per_repeat_label_auc[:, column].mean()) if column in defined else math.nan
        for column, label in enumerate(LABEL_COLUMNS)
    }
    return CVResult(
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
