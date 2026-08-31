import pandas as pd
import pytest

from knee.paths import paths
from knee.series import (
    FAT_SUPPRESSION_COLUMN,
    FLUID_SENSITIVE_COLUMN,
    PLANE_COLUMN,
    SERIES_ID_COLUMN,
    SeriesType,
    classify_series,
    select_series,
)

TRAIN_SERIES_CSV = paths.raw / "train_series.csv"


def test_every_plane_and_contrast_maps_to_a_distinct_type() -> None:
    """Catches the bug where two metadata combos collapse onto one type, silently
    feeding the model a coronal series where selection logic asked for a sagittal."""
    types = {
        classify_series(plane, fluid)
        for plane in ("Sagittal", "Coronal", "Axial")
        for fluid in (0, 1)
    }
    assert types == set(SeriesType)


def test_unknown_plane_is_rejected() -> None:
    """Catches the bug where a metadata contract change at test time (a new or renamed
    plane value) is silently mapped to some default instead of failing fast."""
    with pytest.raises(ValueError, match="Oblique"):
        classify_series("Oblique", 1)


@pytest.mark.skipif(not TRAIN_SERIES_CSV.is_file(), reason="competition data not downloaded")
def test_train_series_metadata_matches_the_assumptions_baked_into_series_type() -> None:
    """SeriesType ignores Fat_Suppression because it duplicates Fluid_Sensitive on
    every train row (the organizers warn test data may diverge; the schema tolerates
    that). Catches a train-data refresh breaking the equality or adding a plane —
    either would mean revisiting series-selection assumptions."""
    import pandas as pd

    series = pd.read_csv(TRAIN_SERIES_CSV)
    assert (series[FLUID_SENSITIVE_COLUMN] == series[FAT_SUPPRESSION_COLUMN]).all()
    observed = {
        classify_series(row[PLANE_COLUMN], row[FLUID_SENSITIVE_COLUMN])
        for _, row in series.drop_duplicates([PLANE_COLUMN, FLUID_SENSITIVE_COLUMN]).iterrows()
    }
    assert observed == set(SeriesType)


def _series_frame(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    """Rows are (series_uid, plane, fluid_sensitive, fat_suppression)."""
    return pd.DataFrame(
        rows,
        columns=[SERIES_ID_COLUMN, PLANE_COLUMN, FLUID_SENSITIVE_COLUMN, FAT_SUPPRESSION_COLUMN],
    )


def test_fluid_sensitive_sagittal_is_preferred() -> None:
    """Catches the bug where selection grabs an arbitrary series — the model would
    train on fluid-sensitive sagittals but score other types at test time."""
    frame = _series_frame(
        [
            ("uid_axial", "Axial", 1, 1),
            ("uid_sag_fluid", "Sagittal", 1, 1),
            ("uid_sag_plain", "Sagittal", 0, 0),
        ]
    )
    assert select_series(frame) == "uid_sag_fluid"


def test_fallback_follows_fluid_first_preference_order() -> None:
    """Catches the bug where a study without the preferred type (6% lack a fluid
    sagittal) falls back to a non-fluid series while fluid coronal/axial exist."""
    frame = _series_frame(
        [
            ("uid_sag_plain", "Sagittal", 0, 0),
            ("uid_cor_fluid", "Coronal", 1, 1),
        ]
    )
    assert select_series(frame) == "uid_cor_fluid"


def test_train_like_flag_combination_wins_within_a_type() -> None:
    """Catches the bug where a test-time series whose flags diverge (organizers warn
    they can) is chosen over the variant matching every train row — a domain shift
    the model never saw."""
    frame = _series_frame(
        [
            ("uid_a_diverged", "Sagittal", 1, 0),
            ("uid_b_trainlike", "Sagittal", 1, 1),
        ]
    )
    assert select_series(frame) == "uid_b_trainlike"


def test_duplicate_types_resolve_deterministically() -> None:
    """Catches nondeterministic selection among duplicate series types — the same
    study would yield different inputs across runs, making experiments incomparable."""
    frame = _series_frame(
        [
            ("uid_b", "Sagittal", 1, 1),
            ("uid_a", "Sagittal", 1, 1),
        ]
    )
    assert select_series(frame) == "uid_a"
    assert select_series(frame.iloc[::-1]) == "uid_a"


def test_empty_study_is_rejected() -> None:
    """Catches a study/series join bug (e.g. UID dtype mismatch) surfacing as a
    confusing IndexError instead of a clear message."""
    with pytest.raises(ValueError, match="no series rows"):
        select_series(_series_frame([]))
