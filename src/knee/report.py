import base64
import html
import io
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict

from knee.data import gold_studies, load_blended_labels
from knee.dicom import DicomDecodeError, load_volume
from knee.fitting import per_label_auc
from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.model import LoadedMultiPlaneModel, load_multiplane_model
from knee.series import TRAIN_SERIES_DIR, best_series_of_type

_STRONG_DISAGREEMENT = 0.4  # |model - miner| beyond this counts as a strong disagreement


class StudyReport(BaseModel):
    """Everything the HTML needs for one study."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    study_uid: str
    is_gold: bool
    probabilities: np.ndarray  # (12,) model
    miner: np.ndarray  # (12,) blended soft labels
    gold: np.ndarray | None  # (12,) 0/1 when the study is gold-labeled
    # label -> list of (item description, weight); the "where it looked" signal.
    attention: dict[str, list[tuple[str, float]]]
    thumbnails: dict[str, str]  # plane value -> data URI of the center slice


def _thumbnail(volume: torch.Tensor) -> str:
    """Center slice of a decoded volume as a PNG data URI."""
    center = (volume[volume.shape[0] // 2].numpy() * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(center).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _predict_with_attention(
    loaded: LoadedMultiPlaneModel, volumes: dict[str, torch.Tensor]
) -> tuple[np.ndarray, dict[str, list[tuple[str, float]]]]:
    """Model probabilities plus per-label attention over the study's bag items."""
    from knee.model import sample_triplets
    from knee.series import SeriesType

    model = loaded.model
    images: list[torch.Tensor] = []
    plane_indices: list[int] = []
    item_names: list[str] = []
    for plane_value, volume in volumes.items():
        series_type = SeriesType(plane_value)
        triplets = sample_triplets(volume, n_anchors=model.n_anchors)
        images.append(triplets)
        index = model.series_types.index(series_type)
        plane_indices.extend([index] * triplets.shape[0])
        item_names.extend([f"{plane_value}#{a}" for a in range(triplets.shape[0])])
    bag = torch.cat(images)
    indices = torch.tensor(plane_indices, dtype=torch.long)
    with torch.no_grad():
        features = model.bag_features(bag, indices)
        logits = model.head(features)
        weights = model.head.attention_weights(features)  # (12, n_items)
    probabilities = torch.sigmoid(logits).numpy()
    attention: dict[str, list[tuple[str, float]]] = {}
    for row, label in enumerate(LABEL_COLUMNS):
        pairs = [(name, float(weights[row, item])) for item, name in enumerate(item_names)]
        pairs.sort(key=lambda pair: -pair[1])
        attention[label] = pairs[:4]
    return probabilities, attention


def _audit_buckets(gold: np.ndarray, miner: np.ndarray, model: np.ndarray) -> dict[str, int]:
    """The three-bucket gold audit over all cells of the gold studies.

    "model caught label error" = the model agrees with gold where the miner does
    not — evidence the labels lever, not the model lever, is losing those points.
    """
    gold_binary = gold >= 0.5
    model_right = (model >= 0.5) == gold_binary
    miner_right = (miner >= 0.5) == gold_binary
    return {
        "both right": int((model_right & miner_right).sum()),
        "model caught label error": int((model_right & ~miner_right).sum()),
        "model error (labels fine)": int((~model_right & miner_right).sum()),
        "both wrong": int((~model_right & ~miner_right).sum()),
    }


def _metric_table(title: str, scores: dict[str, float]) -> str:
    defined = [value for value in scores.values() if not math.isnan(value)]
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{'' if math.isnan(auc) else f'{auc:.3f}'}</td></tr>"
        for label, auc in scores.items()
    )
    macro = f"{float(np.mean(defined)):.3f}" if defined else "n/a"
    return (
        f"<h3>{html.escape(title)} (macro {macro})</h3>"
        f"<table><tr><th>label</th><th>AUC</th></tr>{rows}</table>"
    )


def _study_section(report: StudyReport) -> str:
    thumbs = "".join(
        f'<figure><img src="{uri}" width="160"><figcaption>{html.escape(plane)}</figcaption></figure>'
        for plane, uri in report.thumbnails.items()
    )
    rows: list[str] = []
    for column, label in enumerate(LABEL_COLUMNS):
        gold_cell = "" if report.gold is None else f"{float(report.gold[column]):.0f}"
        top = ", ".join(f"{name} {weight:.2f}" for name, weight in report.attention[label][:2])
        flag = ""
        if report.gold is not None:
            model_right = (report.probabilities[column] >= 0.5) == (report.gold[column] >= 0.5)
            miner_right = (report.miner[column] >= 0.5) == (report.gold[column] >= 0.5)
            if model_right and not miner_right:
                flag = "model caught label error"
            elif not model_right and miner_right:
                flag = "model error"
            elif not model_right:
                flag = "both wrong"
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{report.probabilities[column]:.2f}</td>"
            f"<td>{report.miner[column]:.2f}</td><td>{gold_cell}</td>"
            f"<td>{html.escape(top)}</td><td>{html.escape(flag)}</td></tr>"
        )
    kind = "gold" if report.is_gold else "holdout sample"
    return (
        f'<section id="{html.escape(report.study_uid)}"><h3>{html.escape(report.study_uid)} ({kind})</h3>'
        f"<div class=thumbs>{thumbs}</div>"
        f"<table><tr><th>label</th><th>model</th><th>miner</th><th>gold</th>"
        f"<th>top attention</th><th>verdict</th></tr>{''.join(rows)}</table></section>"
    )



def build_error_report(
    comp_root: Path,
    blended_csv: Path,
    checkpoint: Path,
    out_path: Path,
    *,
    sample: int = 60,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> Path:
    """Score a multiplane checkpoint over the gold studies (+ a sample) and render
    the error-analysis HTML: the gold audit buckets, model/miner/gold AUC tables,
    and per-study panels with per-label attention.

    Args:
        comp_root: Root containing `train.csv`, `train_series.csv`, `train_series/`.
        blended_csv: The blended soft-labels CSV.
        checkpoint: A multiplane checkpoint (`save_multiplane_model`).
        out_path: Destination HTML. Contains StudyInstanceUIDs — keep it out of
            version control (competition rule 2.4.b).
        sample: Non-gold studies to sample alongside the gold set.
        seed: Sampling seed.
        log: Progress sink.

    Returns:
        `out_path`.

    Raises:
        ValueError: Propagated from loading when the checkpoint or labels are invalid.
    """
    loaded = load_multiplane_model(checkpoint)
    loaded.model.eval()

    blended = load_blended_labels(blended_csv)
    miner = blended[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    uid_to_row = {str(uid): row for row, uid in enumerate(blended[STUDY_ID_COLUMN])}

    train_df = pd.read_csv(comp_root / "train.csv")
    gold = gold_studies(train_df)
    gold_labels = {
        str(row[STUDY_ID_COLUMN]): row[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
        for _, row in gold.iterrows()
    }

    rng = np.random.default_rng(seed)
    non_gold = [uid for uid in uid_to_row if uid not in gold_labels]
    sampled = list(rng.choice(non_gold, size=min(sample, len(non_gold)), replace=False))
    study_uids = list(gold_labels) + sampled

    series_df = pd.read_csv(comp_root / "train_series.csv")
    reports: list[StudyReport] = []
    for position, study_uid in enumerate(study_uids, start=1):
        study_series = series_df[series_df[STUDY_ID_COLUMN] == study_uid]
        volumes: dict[str, torch.Tensor] = {}
        for series_type in loaded.series_types:
            series_uid = best_series_of_type(study_series, series_type)
            if series_uid is None:
                continue
            try:
                volumes[series_type.value] = load_volume(
                    comp_root / TRAIN_SERIES_DIR / study_uid / series_uid,
                    size=loaded.input_size,
                    crop_mm=loaded.crop_mm,
                    # Match the checkpoint's training frame — auditing a
                    # mirror-trained model on raw volumes corrupts every
                    # left-knee study and invalidates the gold buckets.
                    canonicalize_laterality=loaded.laterality_normalized,
                )
            except (ValueError, DicomDecodeError):
                continue
        if not volumes:
            continue
        probabilities, attention = _predict_with_attention(loaded, volumes)
        reports.append(
            StudyReport(
                study_uid=study_uid,
                is_gold=study_uid in gold_labels,
                probabilities=probabilities,
                miner=miner[uid_to_row[study_uid]],
                gold=gold_labels.get(study_uid),
                attention=attention,
                thumbnails={plane: _thumbnail(volume) for plane, volume in volumes.items()},
            )
        )
        if position % 20 == 0:
            log(f"scored {position}/{len(study_uids)} studies")

    model_matrix = np.stack([report.probabilities for report in reports])
    miner_matrix = np.stack([report.miner for report in reports])
    vs_miner = per_label_auc((miner_matrix >= 0.5).astype(np.float32), model_matrix)

    gold_reports = [report for report in reports if report.gold is not None]
    gold_matrix = np.stack([report.gold for report in gold_reports if report.gold is not None])
    gold_model = np.stack([report.probabilities for report in gold_reports])
    gold_miner = np.stack([report.miner for report in gold_reports])
    vs_gold = per_label_auc(gold_matrix, gold_model)
    miner_vs_gold = per_label_auc(gold_matrix, gold_miner)
    buckets = _audit_buckets(gold_matrix, gold_miner, gold_model)

    bucket_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in buckets.items())
    body = (
        f"<h1>Error report — {html.escape(checkpoint.name)}</h1>"
        f"<p>{len(reports)} studies scored ({len(gold_reports)} gold, {len(reports) - len(gold_reports)} sampled).</p>"
        f"<h2>The gold audit (all cells of the gold studies)</h2>"
        f"<table><tr><th>bucket</th><th>cells</th></tr>{bucket_rows}</table>"
        f"<p>A large \"model caught label error\" bucket means the LABELS lever is where the points are; "
        f"a large \"model error (labels fine)\" bucket means keep pulling MODEL levers.</p>"
        + _metric_table(f"Model vs gold truth (n={len(gold_reports)} gold studies)", vs_gold)
        + _metric_table("Miner vs gold truth (the teacher's own score)", miner_vs_gold)
        + _metric_table("Model vs miner labels (full evaluated set)", vs_miner)
        + "<h2>Studies</h2>"
        + "".join(_study_section(report) for report in reports if report.is_gold)
    )
    style = (
        "<style>body{font:14px/1.5 system-ui;margin:2rem;max-width:70rem}"
        "table{border-collapse:collapse;margin:0.5rem 0}td,th{border:1px solid #ccc;padding:0.2rem 0.6rem;"
        "text-align:left}.thumbs{display:flex;gap:0.5rem}figure{margin:0}figcaption{font-size:0.8rem}</style>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"<!doctype html><meta charset='utf-8'><title>knee error report</title>{style}{body}")
    log(f"report -> {out_path}")
    return out_path
