from enum import StrEnum

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
