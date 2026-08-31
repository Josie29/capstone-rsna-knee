import os
from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):
    """Filesystem layout for the project.

    Every path is derived from a single root so the same code runs against a local
    checkout, a rented GPU box, and a Kaggle notebook (where the data lives under
    /kaggle/input and the working tree is read-only).
    """

    root: Path
    raw: Path
    interim: Path
    processed: Path
    checkpoints: Path
    submissions: Path

    @classmethod
    def resolve(cls) -> "Paths":
        """Build the layout, honoring KNEE_DATA_ROOT and Kaggle's runtime paths.

        Returns:
            Paths anchored at the repo root, with the data directories redirected to
            KNEE_DATA_ROOT when it is set (a mounted volume on a GPU host) or to the
            Kaggle input/working directories when running inside a Kaggle notebook.
        """
        root = Path(__file__).resolve().parents[2]

        override = os.environ.get("KNEE_DATA_ROOT")
        if override:
            data_root = Path(override)
            work_root = data_root
        elif Path("/kaggle/input").is_dir():
            # Kaggle mounts the competition data read-only; only /kaggle/working is writable.
            data_root = Path("/kaggle/input")
            work_root = Path("/kaggle/working")
        else:
            data_root = root / "data"
            work_root = root

        return cls(
            root=root,
            raw=data_root / "raw",
            interim=data_root / "interim",
            processed=root / "data" / "processed",
            checkpoints=work_root / "checkpoints",
            submissions=work_root / "submissions",
        )


paths = Paths.resolve()
