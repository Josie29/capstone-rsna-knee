import argparse
import base64
import html
import io
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydantic import BaseModel

from knee.labels import LABEL_COLUMNS, STUDY_ID_COLUMN
from knee.paths import paths

MAX_EDGE_PX = 512  # slices are downscaled to this so the HTML stays a reasonable size


class SeriesPanel(BaseModel):
    """One series rendered for the viewer."""

    series_uid: str
    description: str
    rows: int
    columns: int
    slice_data_uris: list[str]


class StudyContext(BaseModel):
    """What train.csv knows about a study, if anything."""

    labels: dict[str, int] | None
    report: str | None


def window_slice(ds: pydicom.Dataset) -> np.ndarray:
    """Apply DICOM windowing to a slice and scale it to 8-bit.

    Args:
        ds: A decoded DICOM slice with pixel data.

    Returns:
        The slice as a uint8 array in [0, 255], windowed with the file's
        WindowCenter/WindowWidth when present, else a 1st-99th percentile stretch.
    """
    pixels = ds.pixel_array.astype(np.float32)
    center = ds.get("WindowCenter")
    width = ds.get("WindowWidth")
    if center is not None and width is not None:
        # Both tags may be multi-valued; the first pair is the default window.
        low = float(np.atleast_1d(center)[0]) - float(np.atleast_1d(width)[0]) / 2
        high = low + float(np.atleast_1d(width)[0])
    else:
        low, high = np.percentile(pixels, [1, 99])
    if high <= low:
        high = low + 1
    scaled = np.clip((pixels - low) / (high - low), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def render_series(series_dir: Path) -> SeriesPanel:
    """Render every slice of one series into embedded PNG data URIs.

    Args:
        series_dir: Directory holding the series' .dcm files.

    Returns:
        The series with slices sorted by InstanceNumber, each a base64 PNG data URI.

    Raises:
        ValueError: If the directory contains no .dcm files.
    """
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise ValueError(f"No .dcm files in {series_dir}")

    # pydicom's dcmread signature has an untyped **kwargs, tripping strict mode.
    slices = [pydicom.dcmread(f) for f in files]  # pyright: ignore[reportUnknownMemberType]
    slices.sort(key=lambda d: int(d.get("InstanceNumber", 0)))

    uris: list[str] = []
    for ds in slices:
        image = Image.fromarray(window_slice(ds))
        if max(image.size) > MAX_EDGE_PX:
            scale = MAX_EDGE_PX / max(image.size)
            image = image.resize((round(image.width * scale), round(image.height * scale)))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        uris.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())

    first = slices[0]
    return SeriesPanel(
        series_uid=series_dir.name,
        description=str(first.get("SeriesDescription", "?")),
        rows=int(first.Rows),
        columns=int(first.Columns),
        slice_data_uris=uris,
    )


def load_study_context(study_uid: str) -> StudyContext:
    """Look a study up in train.csv for labels and report text.

    Args:
        study_uid: The StudyInstanceUID to look up.

    Returns:
        Labels (only if that row is one of the labeled ones) and report text, both
        None when train.csv is absent or the study is not in it (i.e. a test study).
    """
    train_csv = paths.raw / "train.csv"
    if not train_csv.is_file():
        return StudyContext(labels=None, report=None)

    train = pd.read_csv(train_csv)
    match = train[train[STUDY_ID_COLUMN] == study_uid]
    if match.empty:
        return StudyContext(labels=None, report=None)

    row = match.iloc[0]
    labels = None
    if not pd.isna(row[LABEL_COLUMNS[0]]):
        labels = {name: int(row[name]) for name in LABEL_COLUMNS}
    report = None if pd.isna(row["Report"]) else str(row["Report"])
    return StudyContext(labels=labels, report=report)


def build_html(study_uid: str, panels: list[SeriesPanel], context: StudyContext) -> str:
    """Assemble the self-contained viewer page.

    Args:
        study_uid: The study being viewed, shown in the header.
        panels: Rendered series in display order.
        context: Labels/report from train.csv, when available.

    Returns:
        A complete HTML document with all slices embedded.
    """
    if context.labels is not None:
        chips = "".join(
            f'<span class="chip {"pos" if v else "neg"}">{html.escape(k)}: {v}</span>'
            for k, v in context.labels.items()
        )
        label_block = f'<div class="chips">{chips}</div>'
    elif context.report is not None:
        label_block = '<p class="meta">In train.csv but unlabeled (report text only).</p>'
    else:
        label_block = '<p class="meta">Not in train.csv — test study, no labels or report.</p>'

    report_block = ""
    if context.report is not None:
        report_block = (
            "<details><summary>Radiology report</summary>"
            f"<p class='report'>{html.escape(context.report)}</p></details>"
        )

    sections = "".join(
        f"""
        <section class="panel" data-count="{len(p.slice_data_uris)}">
          <h2>{html.escape(p.description)}
            <span class="meta">{len(p.slice_data_uris)} slices · {p.rows}×{p.columns}
             · ...{html.escape(p.series_uid[-12:])}</span></h2>
          <img src="{p.slice_data_uris[len(p.slice_data_uris) // 2]}" alt="{html.escape(p.description)}">
          <div class="controls">
            <input type="range" min="0" max="{len(p.slice_data_uris) - 1}"
                   value="{len(p.slice_data_uris) // 2}">
            <span class="pos-label">{len(p.slice_data_uris) // 2 + 1}/{len(p.slice_data_uris)}</span>
          </div>
          <script type="application/json">{json.dumps(p.slice_data_uris)}</script>
        </section>"""
        for p in panels
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Study ...{html.escape(study_uid[-12:])}</title>
<style>
  :root {{
    --bg: #0d0f12; --panel: #16191f; --text: #d7dce2; --dim: #7c8592;
    --pos: #b3452f; --neg: #2f4d38;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 1rem; background: var(--bg); color: var(--text);
         font: 0.875rem/1.5 system-ui, sans-serif; }}
  h1 {{ font-size: 1.125rem; margin: 0 0 0.25rem; }}
  h2 {{ font-size: 0.9375rem; margin: 0 0 0.5rem; }}
  .meta {{ color: var(--dim); font-weight: normal; font-size: 0.8125rem; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 0.375rem; margin: 0.5rem 0; }}
  .chip {{ padding: 0.125rem 0.5rem; border-radius: 0.75rem; font-size: 0.75rem; }}
  .chip.pos {{ background: var(--pos); }}
  .chip.neg {{ background: var(--neg); }}
  details {{ margin: 0.5rem 0 1rem; }}
  .report {{ white-space: pre-wrap; max-width: 60rem; color: var(--dim); }}
  main {{ display: grid; gap: 1rem; grid-template-columns: 1fr; }}
  @media (min-width: 48rem) {{ main {{ grid-template-columns: repeat(2, 1fr); }} }}
  @media (min-width: 80rem) {{ main {{ grid-template-columns: repeat(3, 1fr); }} }}
  .panel {{ background: var(--panel); border-radius: 0.5rem; padding: 0.75rem; }}
  .panel img {{ width: 100%; max-width: 100%; display: block; border-radius: 0.25rem; }}
  .controls {{ display: flex; align-items: center; gap: 0.5rem; margin-top: 0.5rem; }}
  .controls input {{ flex: 1; }}
  .pos-label {{ color: var(--dim); min-width: 3.5rem; text-align: right; }}
</style>
</head>
<body>
<h1>Study <span class="meta">{html.escape(study_uid)}</span></h1>
{label_block}
{report_block}
<p class="meta">Scroll-wheel over an image, or drag the slider, to move through slices.</p>
<main>{sections}</main>
<script>
  for (const panel of document.querySelectorAll(".panel")) {{
    const uris = JSON.parse(panel.querySelector("script").textContent);
    const img = panel.querySelector("img");
    const slider = panel.querySelector("input");
    const posLabel = panel.querySelector(".pos-label");
    const show = (i) => {{
      const idx = Math.min(Math.max(i, 0), uris.length - 1);
      slider.value = idx;
      img.src = uris[idx];
      posLabel.textContent = `${{idx + 1}}/${{uris.length}}`;
    }};
    slider.addEventListener("input", () => show(Number(slider.value)));
    img.addEventListener("wheel", (e) => {{
      e.preventDefault();
      show(Number(slider.value) + Math.sign(e.deltaY));
    }}, {{ passive: false }});
  }}
</script>
</body>
</html>"""


def find_default_study() -> Path:
    """Locate a study directory when none was given on the command line.

    Returns:
        The first study directory found under data/raw's series folders.

    Raises:
        SystemExit: If no downloaded study exists yet.
    """
    for split in ("train_series", "test_series"):
        split_dir = paths.raw / split
        if split_dir.is_dir():
            studies = sorted(d for d in split_dir.iterdir() if d.is_dir())
            if studies:
                return studies[0]
    raise SystemExit(
        f"No study directories under {paths.raw}/train_series or test_series. "
        "Download one first."
    )


def main() -> None:
    """Render a downloaded study to a self-contained HTML viewer and open it.

    Raises:
        SystemExit: If the requested study directory does not exist.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "study_dir",
        nargs="?",
        type=Path,
        help="Path to a <StudyUID> directory. Default: first study found under data/raw.",
    )
    parser.add_argument("--no-open", action="store_true", help="Skip opening the browser.")
    args = parser.parse_args()

    study_dir: Path = args.study_dir if args.study_dir else find_default_study()
    if not study_dir.is_dir():
        raise SystemExit(f"Not a directory: {study_dir}")

    study_uid = study_dir.name
    series_dirs = sorted(d for d in study_dir.iterdir() if d.is_dir())
    print(f"Rendering {len(series_dirs)} series from {study_uid}", file=sys.stderr)
    panels = [render_series(d) for d in series_dirs]
    context = load_study_context(study_uid)

    out_dir = paths.interim / "viewer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{study_uid}.html"
    out_path.write_text(build_html(study_uid, panels, context))
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)

    if not args.no_open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
