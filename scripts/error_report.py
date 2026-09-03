import argparse
from pathlib import Path

from knee.paths import paths
from knee.report import build_error_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Error-analysis report for a multiplane checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=60, help="non-gold studies to sample")
    parser.add_argument("--out", type=Path, default=paths.interim / "error_report.html")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # paths.processed pins to the repo root regardless of KNEE_DATA_ROOT; derive it
    # from the same data root as `raw` so worktree checkouts resolve correctly.
    build_error_report(
        paths.raw,
        paths.raw.parent / "processed" / "blended_labels_v1.csv",
        args.checkpoint,
        args.out,
        sample=args.sample,
        seed=args.seed,
    )
    print("(report contains StudyInstanceUIDs; lives under data/, never commit)")


if __name__ == "__main__":
    main()
