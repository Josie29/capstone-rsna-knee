import pytest

from knee.paths import paths
from knee.series import (
    FAT_SUPPRESSION_COLUMN,
    FLUID_SENSITIVE_COLUMN,
    PLANE_COLUMN,
    SeriesType,
    classify_series,
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
