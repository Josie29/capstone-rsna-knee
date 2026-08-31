from enum import StrEnum

import pandas as pd

from knee.labels import AnatomicalPlane

# Column names in train_series.csv / test_series.csv.
SERIES_ID_COLUMN = "SeriesInstanceUID"
PLANE_COLUMN = "Anatomical_Plane"
FLUID_SENSITIVE_COLUMN = "Fluid_Sensitive"
FAT_SUPPRESSION_COLUMN = "Fat_Suppression"


class SeriesType(StrEnum):
    """Canonical series vocabulary: plane x fluid-sensitivity, the only 6 combos in
    the training data. Site-dependent SeriesDescription strings are not part of the
    type, and neither is Fat_Suppression: it matches Fluid_Sensitive on every train
    row, and the organizers warn they may diverge at test time, so it is a selection
    tie-breaker (prefer fat-sat among fluid-sensitive candidates), not a type axis.
    """

    SAGITTAL_FLUID = "sagittal_fluid"
    SAGITTAL_NONFLUID = "sagittal_nonfluid"
    CORONAL_FLUID = "coronal_fluid"
    CORONAL_NONFLUID = "coronal_nonfluid"
    AXIAL_FLUID = "axial_fluid"
    AXIAL_NONFLUID = "axial_nonfluid"


_TYPE_BY_PLANE_AND_FLUID: dict[tuple[AnatomicalPlane, bool], SeriesType] = {
    (AnatomicalPlane.SAGITTAL, True): SeriesType.SAGITTAL_FLUID,
    (AnatomicalPlane.SAGITTAL, False): SeriesType.SAGITTAL_NONFLUID,
    (AnatomicalPlane.CORONAL, True): SeriesType.CORONAL_FLUID,
    (AnatomicalPlane.CORONAL, False): SeriesType.CORONAL_NONFLUID,
    (AnatomicalPlane.AXIAL, True): SeriesType.AXIAL_FLUID,
    (AnatomicalPlane.AXIAL, False): SeriesType.AXIAL_NONFLUID,
}


def classify_series(plane: str, fluid_sensitive: int | bool) -> SeriesType:
    """Map one train_series.csv/test_series.csv row to its canonical series type.

    Args:
        plane: The `Anatomical_Plane` value ("Sagittal", "Coronal", or "Axial").
        fluid_sensitive: The `Fluid_Sensitive` value (0/1 in the CSVs).

    Returns:
        The canonical type for this series.

    Raises:
        ValueError: If `plane` is not a known anatomical plane. Raised rather than
            defaulted because an unknown plane at test time means the metadata
            contract changed and series selection logic cannot be trusted.
    """
    try:
        validated_plane = AnatomicalPlane(plane)
    except ValueError as exc:
        raise ValueError(
            f"Unknown Anatomical_Plane {plane!r}; expected one of "
            f"{[p.value for p in AnatomicalPlane]}"
        ) from exc
    return _TYPE_BY_PLANE_AND_FLUID[(validated_plane, bool(fluid_sensitive))]


# Fluid-sensitive series carry the pathology signal (effusion, edema, tears light up),
# so all three fluid types outrank any non-fluid one; sagittal first because 94% of
# train studies have a fluid-sensitive sagittal and it covers ACL/meniscus best.
SERIES_TYPE_PREFERENCE: tuple[SeriesType, ...] = (
    SeriesType.SAGITTAL_FLUID,
    SeriesType.CORONAL_FLUID,
    SeriesType.AXIAL_FLUID,
    SeriesType.SAGITTAL_NONFLUID,
    SeriesType.CORONAL_NONFLUID,
    SeriesType.AXIAL_NONFLUID,
)


def select_series(study_series: pd.DataFrame) -> str:
    """Pick the single series to feed the model for one study.

    Studies average 5.5 series and can carry duplicates of a type, so selection must
    be deterministic. Preference is `SERIES_TYPE_PREFERENCE`; within a type, series
    where `Fat_Suppression == Fluid_Sensitive` win (the variant matching every train
    row — the organizers warn the flags can diverge on test data), then the
    lexicographically smallest SeriesInstanceUID breaks remaining ties.

    Args:
        study_series: Rows of train_series.csv/test_series.csv for ONE study; must
            contain the series UID, plane, and both contrast-flag columns.

    Returns:
        The chosen SeriesInstanceUID.

    Raises:
        ValueError: If `study_series` is empty or contains an unknown plane (via
            `classify_series`) — every study in the data has all six-type coverage
            assumptions checked upstream, so an empty frame means a join bug.
    """
    if study_series.empty:
        raise ValueError("select_series called with no series rows for the study")

    rank = {series_type: index for index, series_type in enumerate(SERIES_TYPE_PREFERENCE)}

    def preference(row: pd.Series) -> tuple[int, bool, str]:
        # Lower is better on every key: preference rank, then non-train-like flag
        # combos pushed after train-like ones, then UID for determinism.
        series_type = classify_series(row[PLANE_COLUMN], row[FLUID_SENSITIVE_COLUMN])
        flags_diverge = bool(row[FAT_SUPPRESSION_COLUMN] != row[FLUID_SENSITIVE_COLUMN])
        return rank[series_type], flags_diverge, str(row[SERIES_ID_COLUMN])

    best = min((row for _, row in study_series.iterrows()), key=preference)
    return str(best[SERIES_ID_COLUMN])
