from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from pydantic import BaseModel
from torchvision.transforms import InterpolationMode

from knee.cv import EVAL_THRESHOLD
from knee.dicom import DicomDecodeError, load_volume
from knee.fitting import per_label_auc, positive_weight, weighted_bce
from knee.infer import merge_predictions
from knee.labels import STUDY_ID_COLUMN
from knee.model import (
    HeadType,
    InputMode,
    KneeModel,
    MultiPlaneModel,
    resolve_device,
    sample_triplets,
    save_model,
    save_multiplane_model,
)
from knee.series import TRAIN_SERIES_DIR, SeriesType, best_series_of_type

_CACHE_WORKERS = 4


class CacheBuildResult(BaseModel):
    """Outcome of one pixel-cache build pass."""

    n_studies: int
    # Studies with a cached stack, per plane (mirrors FeatureBank.plane_coverage).
    coverage: dict[SeriesType, int]


class FinetuneConfig(BaseModel):
    """Hyperparameters for one per-plane fine-tune.

    Defaults follow the staged-unfreeze recipe: a short frozen warm-up so the
    (already warm-started) head settles before gradients reach the backbone, then
    discriminative learning rates — the backbone crawls at a tenth of the head's
    pace so pretrained features adapt instead of being trashed.
    """

    epochs: int = 15
    frozen_epochs: int = 2
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_studies: int = 16
    n_anchors: int = 3
    anchor_window: tuple[float, float] = (0.2, 0.8)
    anchor_jitter: float = 0.05
    max_rotation_degrees: float = 10.0
    scale_jitter: float = 0.1
    intensity_jitter: float = 0.1
    seed: int = 0


class FinetuneResult(BaseModel):
    """Outcome of fine-tuning one per-plane specialist."""

    series_type: SeriesType
    n_train: int
    n_val: int
    best_epoch: int
    best_val_macro_auc: float
    val_auc_per_label: dict[str, float]
    checkpoint_path: Path


def cache_path(cache_dir: Path, series_type: SeriesType, row: int) -> Path:
    """Where one study/plane stack lives in the pixel cache (row = labels-frame index)."""
    return cache_dir / series_type.value / f"{row}.npy"


def build_pixel_cache(
    comp_root: Path,
    labels: pd.DataFrame,
    cache_dir: Path,
    *,
    series_types: Sequence[SeriesType],
    input_size: int = 224,
    crop_mm: float | None = None,
    log: Callable[[str], None] = print,
) -> CacheBuildResult:
    """One decode pass writing every sorted/cropped/resized stack as uint8 .npy.

    Fine-tuning needs pixels every epoch; decoding costs ~75 min per pass, so this
    pays it once and epochs read tensors from disk instead. Existing files are kept
    (the build resumes), and studies whose series is missing or fails to decode are
    skipped exactly like `collect_features` — that plane sits out for the study.

    Args:
        comp_root: Competition data root containing `train_series.csv` and
            `train_series/`.
        labels: One row per study: `StudyInstanceUID` plus the 12 label columns.
        cache_dir: Cache root; one subdirectory per plane.
        series_types: Planes to cache.
        input_size: Slice resize target fed to `load_volume`.
        crop_mm: Fixed-mm crop fed to `load_volume` (None = full frame). Bake the
            experiment's geometry in here — the cache IS the training input.
        log: Progress sink (`print` in notebooks).

    Returns:
        Coverage per plane, for the same sanity prints the bank flow uses.
    """
    series_df = pd.read_csv(comp_root / "train_series.csv")
    series_groups: dict[str, pd.DataFrame] = {
        str(uid): group for uid, group in series_df.groupby(STUDY_ID_COLUMN)
    }
    no_series = series_df.iloc[0:0]
    study_uids = [str(uid) for uid in labels[STUDY_ID_COLUMN]]
    for series_type in series_types:
        (cache_dir / series_type.value).mkdir(parents=True, exist_ok=True)

    def cache_one(row: int, series_type: SeriesType) -> bool:
        target = cache_path(cache_dir, series_type, row)
        if target.exists():
            return True
        series_uid = best_series_of_type(series_groups.get(study_uids[row], no_series), series_type)
        if series_uid is None:
            return False
        try:
            volume = load_volume(
                comp_root / TRAIN_SERIES_DIR / study_uids[row] / series_uid,
                size=input_size,
                crop_mm=crop_mm,
            )
        except (ValueError, DicomDecodeError) as exc:
            log(f"{series_type}: skipping series of study {row + 1}: {exc}")
            return False
        np.save(target, (volume.numpy() * 255).astype(np.uint8))
        return True

    coverage = {series_type: 0 for series_type in series_types}
    with ThreadPoolExecutor(max_workers=_CACHE_WORKERS) as executor:
        futures = {
            executor.submit(cache_one, row, series_type): series_type
            for row in range(len(study_uids))
            for series_type in series_types
        }
        for done, future in enumerate(as_completed(futures), start=1):
            if future.result():
                coverage[futures[future]] += 1
            if done % 500 == 0:
                log(f"cached {done}/{len(futures)}")
    return CacheBuildResult(n_studies=len(study_uids), coverage=coverage)


def optimizer_param_groups(module: torch.nn.Module, lr: float, weight_decay: float) -> list[dict[str, object]]:
    """Two AdamW groups: weights get decay; norms and biases don't.

    Decaying normalization scales and biases is a known fine-tuning destabilizer
    (ViTs especially — every block is LayerNorm-coupled); excluding them is the
    standard recipe. Params are split by shape (ndim <= 1 catches norm scales and
    biases), not by requires_grad — the staged unfreeze toggles that after the
    optimizer is built, and frozen params receive no grads anyway.

    Args:
        module: The module whose parameters to group.
        lr: Learning rate for both groups.
        weight_decay: Decay for the weight group; the norm/bias group gets 0.

    Returns:
        Param-group dicts ready for the optimizer.
    """
    decay = [p for p in module.parameters() if p.ndim > 1]
    no_decay = [p for p in module.parameters() if p.ndim <= 1]
    return [
        {"params": decay, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


def _load_stack(path: Path) -> torch.Tensor:
    """One cached uint8 stack back to (n_slices, H, W) float in [0, 1]."""
    return torch.from_numpy(np.load(path).astype(np.float32) / 255.0)  # pyright: ignore[reportUnknownMemberType]


def _augment(images: torch.Tensor, config: FinetuneConfig, rng: np.random.Generator) -> torch.Tensor:
    """One study's triplet images under a shared random rotate/scale/intensity jitter.

    Deliberately no horizontal flip: mirroring a knee swaps medial<->lateral anatomy
    while the labels stay put, silently corrupting 5 of the 12 findings.
    """
    angle = float(rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees))
    scale = float(rng.uniform(1 - config.scale_jitter, 1 + config.scale_jitter))
    out = TF.affine(
        images, angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
        interpolation=InterpolationMode.BILINEAR,
    )
    gain = float(rng.uniform(1 - config.intensity_jitter, 1 + config.intensity_jitter))
    bias = float(rng.uniform(-config.intensity_jitter / 2, config.intensity_jitter / 2))
    return (out * gain + bias).clamp(0.0, 1.0)


def _study_logits(model: KneeModel, features: torch.Tensor) -> torch.Tensor:
    """(batch, K, feat) triplet features -> (batch, 12) logits via the model's head."""
    if model.head_type is HeadType.ATTENTION:
        return model.head(features)  # PerLabelAttentionHead handles batched input
    pooled = torch.cat([features.mean(dim=1), features.max(dim=1).values], dim=-1)
    return model.head(pooled)


def _evaluate(
    model: KneeModel,
    cache_dir: Path,
    series_type: SeriesType,
    rows: list[int],
    binary_targets: np.ndarray,
    config: FinetuneConfig,
    device: str,
) -> dict[str, float]:
    """Per-label validation AUC with deterministic anchors and no augmentation."""
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for row in rows:
            stack = _load_stack(cache_path(cache_dir, series_type, row))
            images = sample_triplets(stack, n_anchors=config.n_anchors, window=config.anchor_window)
            features = model.triplet_features(images.to(device)).unsqueeze(0)
            logits = _study_logits(model, features)
            probabilities.append(torch.sigmoid(logits.float()).squeeze(0).cpu().numpy())
    return per_label_auc(binary_targets, np.stack(probabilities))


def evaluate_ensemble_holdout(
    cache_dir: Path,
    models: dict[SeriesType, KneeModel],
    targets: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, float]:
    """Fine-tuned ensemble AUC on the validation split.

    The paired counterpart to `cv.evaluate_holdout`: same split, same combiner
    (`merge_predictions`, plane sit-outs included), but predictions come from the
    fine-tuned per-plane models via their own deterministic triplet sampling — so
    frozen-vs-finetuned ensemble numbers differ only in the training.

    Args:
        cache_dir: Pixel cache the models were trained from.
        models: One fine-tuned model per plane (e.g. reloaded via `load_model`).
        targets: (n_studies, 12) float labels aligned with the cache's row indices.
        val_mask: (n_studies,) bool, True = validation.

    Returns:
        Label name -> validation AUC vs labels thresholded at `EVAL_THRESHOLD`.
    """
    device = resolve_device()
    for model in models.values():
        model.to(device)
        model.eval()
    series_types = list(models)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for row in np.flatnonzero(val_mask):
            per_model_rows: list[np.ndarray] = []
            for series_type in series_types:
                stack_file = cache_path(cache_dir, series_type, int(row))
                if not stack_file.exists():
                    per_model_rows.append(np.full(targets.shape[1], np.nan))
                    continue
                stack = _load_stack(stack_file).to(device)
                per_model_rows.append(models[series_type].predict_study(stack).cpu().numpy())
            predictions.append(merge_predictions(np.stack(per_model_rows), series_types))
    binary = (targets[val_mask] >= EVAL_THRESHOLD).astype(np.float32)
    return per_label_auc(binary, np.stack(predictions))


def finetune_plane(
    cache_dir: Path,
    targets: np.ndarray,
    val_mask: np.ndarray,
    *,
    series_type: SeriesType,
    model: KneeModel,
    out_path: Path,
    config: FinetuneConfig | None = None,
    input_size: int = 224,
    crop_mm: float | None = None,
    cell_weights: np.ndarray | None = None,
    label_source: str = "blended_v1",
    log: Callable[[str], None] = print,
) -> FinetuneResult:
    """Fine-tune one per-plane specialist end to end on cached 2.5D triplets.

    Staged: `frozen_epochs` of head-only training first (the E005 regime — the
    warm-started head settles), then the backbone unfreezes at a tenth of the
    head's learning rate under a cosine schedule. Every epoch is scored on the
    fixed validation split (deterministic anchors, no augmentation) and the best
    checkpoint by validation macro AUC is what gets saved — full CV is infeasible
    at one-fine-tune-per-fold, so the split IS the protocol (mark it in
    experiments.md).

    Args:
        cache_dir: Pixel cache from `build_pixel_cache` (its crop is the input
            geometry — pass the matching `crop_mm` for checkpoint provenance).
        targets: (n_studies, 12) float labels, aligned with the labels frame the
            cache was built from.
        val_mask: (n_studies,) bool from `stratified_holdout`; True = validation.
        series_type: The plane this specialist trains on.
        model: A `KneeModel` with `input_mode=TRIPLETS`; warm-start its head from
            the best frozen run before calling.
        out_path: Destination checkpoint.
        config: Hyperparameters; defaults when omitted.
        input_size: Stamped into the checkpoint (must match the cache build).
        crop_mm: Stamped into the checkpoint (must match the cache build).
        cell_weights: Optional (n_studies, 12) confidence weights aligned with
            `targets` (see `fitting.weighted_bce`); validation stays unweighted.
        label_source: Provenance string for the checkpoint.
        log: Progress sink (`print` in notebooks).

    Returns:
        The result, with per-label validation AUC of the best epoch.

    Raises:
        ValueError: If the model's input mode is not TRIPLETS, or either split has
            no cached studies for this plane.
    """
    config = config or FinetuneConfig()
    if model.input_mode is not InputMode.TRIPLETS:
        raise ValueError("finetune_plane requires a model built with input_mode=TRIPLETS")

    cached = [row for row in range(targets.shape[0]) if cache_path(cache_dir, series_type, row).exists()]
    train_rows = [row for row in cached if not val_mask[row]]
    val_rows = [row for row in cached if val_mask[row]]
    if not train_rows or not val_rows:
        raise ValueError(f"{series_type}: empty split (train {len(train_rows)}, val {len(val_rows)})")
    binary_val = (targets[val_rows] >= EVAL_THRESHOLD).astype(np.float32)
    if not any(len(np.unique(binary_val[:, column])) == 2 for column in range(binary_val.shape[1])):
        raise ValueError(f"{series_type}: no validation label has both classes; macro AUC would be undefined")
    train_targets = torch.from_numpy(targets[train_rows])  # pyright: ignore[reportUnknownMemberType]

    device = resolve_device()
    model.to(device)
    pos_weight = positive_weight(train_targets).to(device)
    train_weights = (
        torch.from_numpy(cell_weights[train_rows])  # pyright: ignore[reportUnknownMemberType]
        if cell_weights is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model.backbone, config.backbone_lr, config.weight_decay)
        + optimizer_param_groups(model.head, config.head_lr, config.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    # fp16 autocast without a scaler silently underflows small gradients — the
    # scaler is what makes mixed precision safe on a T4 (no bf16 there).
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.default_rng(config.seed)

    best_macro, best_epoch, best_state, best_per_label = -1.0, -1, None, {}
    for epoch in range(config.epochs):
        frozen = epoch < config.frozen_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad = not frozen
        model.train()
        order = [int(i) for i in rng.permutation(len(train_rows))]
        for start in range(0, len(order), config.batch_studies):
            batch_indices = order[start : start + config.batch_studies]
            batch_rows = [train_rows[i] for i in batch_indices]
            images = torch.stack(
                [
                    _augment(
                        sample_triplets(
                            _load_stack(cache_path(cache_dir, series_type, row)),
                            n_anchors=config.n_anchors,
                            window=config.anchor_window,
                            jitter=config.anchor_jitter,
                            rng=rng,
                        ),
                        config,
                        rng,
                    )
                    for row in batch_rows
                ]
            )  # (B, K, 3, H, W)
            flat = images.view(-1, *images.shape[2:]).to(device)
            # train_targets is aligned with train_rows, and `order` indexes train_rows.
            batch_targets = train_targets[torch.as_tensor(batch_indices)].to(device)
            optimizer.zero_grad()
            on_cuda = device == "cuda"
            with torch.autocast("cuda", enabled=on_cuda):
                features = model.triplet_features(flat).view(len(batch_rows), config.n_anchors, -1)
                loss = weighted_bce(
                    _study_logits(model, features.float()),
                    batch_targets,
                    pos_weight=pos_weight,
                    cell_weights=train_weights[torch.as_tensor(batch_indices)].to(device)
                    if train_weights is not None
                    else None,
                )
            scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]
            scaler.step(optimizer)  # pyright: ignore[reportUnknownMemberType]
            scaler.update()
        scheduler.step()

        val_auc = _evaluate(model, cache_dir, series_type, val_rows, binary_val, config, device)
        defined = [value for value in val_auc.values() if not np.isnan(value)]
        macro = float(np.mean(defined)) if defined else float("nan")
        log(f"{series_type.value} epoch {epoch + 1}/{config.epochs}"
            f"{' (frozen)' if frozen else ''}: val macro AUC {macro:.3f}")
        if macro > best_macro:
            best_macro, best_epoch, best_per_label = macro, epoch, val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None  # epochs >= 1, so at least one eval happened
    model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()
    save_model(
        model,
        out_path,
        input_size=input_size,
        series_type=series_type,
        label_source=label_source,
        n_studies=len(train_rows),
        crop_mm=crop_mm,
    )
    log(f"{series_type.value}: best epoch {best_epoch + 1} (val macro {best_macro:.3f}) -> {out_path}")
    return FinetuneResult(
        series_type=series_type,
        n_train=len(train_rows),
        n_val=len(val_rows),
        best_epoch=best_epoch,
        best_val_macro_auc=best_macro,
        val_auc_per_label=best_per_label,
        checkpoint_path=out_path,
    )


class UnifiedFinetuneResult(BaseModel):
    """Outcome of fine-tuning the unified multi-plane model."""

    series_types: list[SeriesType]
    n_train: int
    n_val: int
    best_epoch: int
    best_val_macro_auc: float
    val_auc_per_label: dict[str, float]
    checkpoint_path: Path


def _study_bag(
    cache_dir: Path,
    model: MultiPlaneModel,
    row: int,
    config: FinetuneConfig,
    rng: np.random.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One study's bag: triplets from every cached plane, tagged with plane indices.

    Deterministic anchors when `rng` is None (evaluation); jittered otherwise.
    """
    images: list[torch.Tensor] = []
    plane_indices: list[int] = []
    for plane_index, series_type in enumerate(model.series_types):
        stack_file = cache_path(cache_dir, series_type, row)
        if not stack_file.exists():
            continue  # plane sits out; the bag shrinks
        triplets = sample_triplets(
            _load_stack(stack_file),
            n_anchors=config.n_anchors,
            window=config.anchor_window,
            jitter=config.anchor_jitter if rng is not None else 0.0,
            rng=rng,
        )
        images.append(triplets)
        plane_indices.extend([plane_index] * triplets.shape[0])
    return torch.cat(images), torch.tensor(plane_indices, dtype=torch.long)


def _evaluate_unified(
    model: MultiPlaneModel,
    cache_dir: Path,
    rows: list[int],
    binary_targets: np.ndarray,
    device: str,
) -> dict[str, float]:
    """Validation AUC via each study's deterministic bag (the model IS the ensemble)."""
    model.eval()
    probabilities: list[np.ndarray] = []
    for row in rows:
        volumes = {
            series_type: _load_stack(cache_path(cache_dir, series_type, row))
            for series_type in model.series_types
            if cache_path(cache_dir, series_type, row).exists()
        }
        probabilities.append(model.predict_study(volumes).cpu().numpy())
    return per_label_auc(binary_targets, np.stack(probabilities))


def finetune_unified(
    cache_dir: Path,
    targets: np.ndarray,
    val_mask: np.ndarray,
    *,
    model: MultiPlaneModel,
    out_path: Path,
    config: FinetuneConfig | None = None,
    input_size: int = 224,
    crop_mm: float | None = None,
    cell_weights: np.ndarray | None = None,
    label_source: str = "blended_v1",
    log: Callable[[str], None] = print,
) -> UnifiedFinetuneResult:
    """Fine-tune the unified multi-plane model end to end on cached 2.5D bags.

    The study-level counterpart of `finetune_plane`: per step, each study
    contributes one bag (triplets from every cached plane, plane-embedded), padded
    and masked across the batch. Same staged unfreeze, discriminative LRs, cosine
    schedule, no-flip augmentation, and best-epoch selection; per-epoch validation
    uses each study's deterministic bag, so it IS the holdout ensemble number.

    Args:
        cache_dir: Pixel cache from `build_pixel_cache` (all of the model's planes).
        targets: (n_studies, 12) float labels aligned with the cache row indices.
        val_mask: (n_studies,) bool from `stratified_holdout`; True = validation.
        model: A `MultiPlaneModel`; optionally warm-started before calling.
        out_path: Destination checkpoint (`save_multiplane_model`).
        config: Hyperparameters; defaults when omitted.
        input_size: Stamped into the checkpoint (must match the cache build).
        crop_mm: Stamped into the checkpoint (must match the cache build).
        cell_weights: Optional (n_studies, 12) confidence weights; validation stays
            unweighted.
        label_source: Provenance string for the checkpoint.
        log: Progress sink (`print` in notebooks).

    Returns:
        The result, with per-label validation AUC of the best epoch.

    Raises:
        ValueError: If either split has no study with a cached plane, or no
            validation label has both classes.
    """
    config = config or FinetuneConfig()
    cached = [
        row
        for row in range(targets.shape[0])
        if any(cache_path(cache_dir, t, row).exists() for t in model.series_types)
    ]
    train_rows = [row for row in cached if not val_mask[row]]
    val_rows = [row for row in cached if val_mask[row]]
    if not train_rows or not val_rows:
        raise ValueError(f"empty split (train {len(train_rows)}, val {len(val_rows)})")
    binary_val = (targets[val_rows] >= EVAL_THRESHOLD).astype(np.float32)
    if not any(len(np.unique(binary_val[:, column])) == 2 for column in range(binary_val.shape[1])):
        raise ValueError("no validation label has both classes; macro AUC would be undefined")
    train_targets = torch.from_numpy(targets[train_rows])  # pyright: ignore[reportUnknownMemberType]

    device = resolve_device()
    model.to(device)
    pos_weight = positive_weight(train_targets).to(device)
    train_weights = (
        torch.from_numpy(cell_weights[train_rows])  # pyright: ignore[reportUnknownMemberType]
        if cell_weights is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model.backbone, config.backbone_lr, config.weight_decay)
        + optimizer_param_groups(model.plane_embeddings, config.head_lr, config.weight_decay)
        + optimizer_param_groups(model.head, config.head_lr, config.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    # fp16 autocast without a scaler silently underflows small gradients — the
    # scaler is what makes mixed precision safe on a T4 (no bf16 there).
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.default_rng(config.seed)

    best_macro, best_epoch, best_state, best_per_label = -1.0, -1, None, {}
    for epoch in range(config.epochs):
        frozen = epoch < config.frozen_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad = not frozen
        model.train()
        order = [int(i) for i in rng.permutation(len(train_rows))]
        for start in range(0, len(order), config.batch_studies):
            batch_indices = order[start : start + config.batch_studies]
            bags = [
                _study_bag(cache_dir, model, train_rows[i], config, rng) for i in batch_indices
            ]
            bags = [(_augment(images, config, rng), planes) for images, planes in bags]
            max_items = max(images.shape[0] for images, _ in bags)
            flat = torch.cat([images for images, _ in bags]).to(device)
            flat_planes = torch.cat([planes for _, planes in bags]).to(device)
            optimizer.zero_grad()
            on_cuda = device == "cuda"
            with torch.autocast("cuda", enabled=on_cuda):
                features = model.bag_features(flat, flat_planes).float()
                # Re-pack variable-length bags into a padded batch for the head.
                padded = torch.zeros(len(bags), max_items, features.shape[1], device=device)
                mask = torch.zeros(len(bags), max_items, dtype=torch.bool, device=device)
                offset = 0
                for bag_index, (images, _) in enumerate(bags):
                    count = images.shape[0]
                    padded[bag_index, :count] = features[offset : offset + count]
                    mask[bag_index, :count] = True
                    offset += count
                logits = model.head(padded, mask)
                loss = weighted_bce(
                    logits,
                    train_targets[torch.as_tensor(batch_indices)].to(device),
                    pos_weight=pos_weight,
                    cell_weights=train_weights[torch.as_tensor(batch_indices)].to(device)
                    if train_weights is not None
                    else None,
                )
            scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]
            scaler.step(optimizer)  # pyright: ignore[reportUnknownMemberType]
            scaler.update()
        scheduler.step()

        val_auc = _evaluate_unified(model, cache_dir, val_rows, binary_val, device)
        defined = [value for value in val_auc.values() if not np.isnan(value)]
        macro = float(np.mean(defined)) if defined else float("nan")
        log(f"unified epoch {epoch + 1}/{config.epochs}"
            f"{' (frozen)' if frozen else ''}: val macro AUC {macro:.3f}")
        if macro > best_macro:
            best_macro, best_epoch, best_per_label = macro, epoch, val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None  # the both-classes guard above ensures a finite macro
    model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()
    save_multiplane_model(
        model,
        out_path,
        input_size=input_size,
        label_source=label_source,
        n_studies=len(train_rows),
        crop_mm=crop_mm,
    )
    log(f"unified: best epoch {best_epoch + 1} (val macro {best_macro:.3f}) -> {out_path}")
    return UnifiedFinetuneResult(
        series_types=list(model.series_types),
        n_train=len(train_rows),
        n_val=len(val_rows),
        best_epoch=best_epoch,
        best_val_macro_auc=best_macro,
        val_auc_per_label=best_per_label,
        checkpoint_path=out_path,
    )
