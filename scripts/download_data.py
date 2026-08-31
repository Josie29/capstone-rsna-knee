import os
import shutil
import subprocess
import sys
from pathlib import Path

from knee.paths import paths

COMPETITION = "rsna-knee-abnormality-detection"


def check_kaggle_cli() -> str:
    """Locate the Kaggle CLI and confirm credentials are in place.

    Returns:
        Absolute path to the `kaggle` executable.

    Raises:
        SystemExit: If the CLI is not installed or no credentials were found.
    """
    executable = shutil.which("kaggle")
    if executable is None:
        raise SystemExit(
            "kaggle CLI not found. Install it with:  uv sync --extra data\n"
            "Then run this script through the project venv:  uv run scripts/download_data.py"
        )

    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.is_file() and "KAGGLE_USERNAME" not in os.environ:
        raise SystemExit(
            f"No Kaggle credentials. Create an API token at "
            f"https://www.kaggle.com/settings/account and save it to {token}, "
            f"or export KAGGLE_USERNAME and KAGGLE_KEY."
        )

    return executable


def download(destination: Path) -> None:
    """Download and unzip the competition archive into `destination`.

    Args:
        destination: Directory to unpack into. Created if it does not exist.

    Raises:
        SystemExit: If the Kaggle CLI exits non-zero, most often because the
            competition rules have not been accepted on the website yet.
    """
    executable = check_kaggle_cli()
    destination.mkdir(parents=True, exist_ok=True)

    command = [executable, "competitions", "download", "-c", COMPETITION, "-p", str(destination)]
    print(f"$ {' '.join(command)}", file=sys.stderr)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"kaggle exited {result.returncode}. If this is a 403, accept the competition "
            f"rules at https://www.kaggle.com/competitions/{COMPETITION}/rules first."
        )

    archive = destination / f"{COMPETITION}.zip"
    if archive.is_file():
        print(f"Unzipping {archive} -> {destination}", file=sys.stderr)
        shutil.unpack_archive(archive, destination)
        archive.unlink()

    print(f"Done. Data is in {destination}", file=sys.stderr)


if __name__ == "__main__":
    download(paths.raw)
