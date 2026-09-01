import json
from pathlib import Path

NOTEBOOKS_GLOB = "notebooks/**/*.ipynb"


def normalize_notebook(path: Path) -> bool:
    """Rewrite a notebook into canonical nbformat shape.

    Programmatic cell edits leave `source` as one string; canonical Jupyter files
    store a list of lines. Jupyter executes both, but Kaggle's viewer falls back to
    a raw-JSON display for non-canonical files — so every notebook must be
    normalized before `kaggle kernels push` or the pushed version renders as JSON.

    Args:
        path: The .ipynb file to normalize in place.

    Returns:
        True if the file was modified, False if it was already canonical.

    Raises:
        ValueError: If the file is not valid notebook JSON with a `cells` list.
    """
    try:
        notebook = json.loads(path.read_text())
        cells = notebook["cells"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"{path} is not a valid notebook: {exc}") from exc

    changed = False
    for cell in cells:
        source = cell.get("source", [])
        if isinstance(source, str):
            cell["source"] = source.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return changed


def main() -> None:
    """Normalize every notebook under notebooks/; print what changed."""
    repo_root = Path(__file__).resolve().parent.parent
    for path in sorted(repo_root.glob(NOTEBOOKS_GLOB)):
        state = "normalized" if normalize_notebook(path) else "already canonical"
        print(f"{path.relative_to(repo_root)}: {state}")


if __name__ == "__main__":
    main()
