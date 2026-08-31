from pathlib import Path

import pytest

from knee.paths import Paths


def test_data_root_override_redirects_raw_and_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Catches the bug where a training run on a GPU box with a mounted data volume
    silently writes checkpoints back into the repo checkout and fills the boot disk."""
    monkeypatch.setenv("KNEE_DATA_ROOT", str(tmp_path))

    resolved = Paths.resolve()

    assert resolved.raw == tmp_path / "raw"
    assert resolved.checkpoints == tmp_path / "checkpoints"
    # processed holds small committed manifests, so it stays in the checkout either way.
    assert resolved.processed == resolved.root / "data" / "processed"


def test_default_layout_is_anchored_at_the_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches the bug where importing knee from a subdirectory resolves the data root
    relative to the working directory instead of the process working directory."""
    monkeypatch.delenv("KNEE_DATA_ROOT", raising=False)

    resolved = Paths.resolve()

    assert (resolved.root / "pyproject.toml").is_file()
    assert resolved.raw == resolved.root / "data" / "raw"
