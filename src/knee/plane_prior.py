from collections.abc import Sequence

from knee.labels import Label
from knee.series import SeriesType

# Clinical relevance of each fluid-sensitive plane per finding, graded coarsely:
#   3 = plane of choice, 2 = useful, 1 = limited.
# Interim fixed combiner weights until the validation split enables learned ones
# (DECISIONS.md #3). Sources and rubric:
# docs/clinical understanding/plane-abnormality-relevance.md
PLANE_PRIOR: dict[SeriesType, dict[Label, int]] = {
    SeriesType.SAGITTAL_FLUID: {
        Label.ACL: 3,
        Label.MCL: 1,
        Label.MEDIAL_MENISCUS: 3,
        Label.LATERAL_MENISCUS: 3,
        Label.MEDIAL_OA: 2,
        Label.LATERAL_OA: 2,
        Label.PATELLOFEMORAL_OA: 2,
        Label.EFFUSION: 2,
        Label.SYNOVITIS: 2,
        Label.BAKERS_CYST: 2,
        Label.CONTUSION: 2,
        Label.FRACTURE: 2,
    },
    SeriesType.CORONAL_FLUID: {
        Label.ACL: 2,
        Label.MCL: 3,
        Label.MEDIAL_MENISCUS: 3,
        Label.LATERAL_MENISCUS: 3,
        Label.MEDIAL_OA: 3,
        Label.LATERAL_OA: 3,
        Label.PATELLOFEMORAL_OA: 1,
        Label.EFFUSION: 1,
        Label.SYNOVITIS: 1,
        Label.BAKERS_CYST: 1,
        Label.CONTUSION: 2,
        Label.FRACTURE: 2,
    },
    SeriesType.AXIAL_FLUID: {
        Label.ACL: 2,
        Label.MCL: 2,
        Label.MEDIAL_MENISCUS: 1,
        Label.LATERAL_MENISCUS: 1,
        Label.MEDIAL_OA: 1,
        Label.LATERAL_OA: 1,
        Label.PATELLOFEMORAL_OA: 3,
        Label.EFFUSION: 3,
        Label.SYNOVITIS: 3,
        Label.BAKERS_CYST: 3,
        Label.CONTUSION: 2,
        Label.FRACTURE: 2,
    },
}


def combiner_weights(series_types: Sequence[SeriesType], label: Label) -> tuple[float, ...]:
    """Normalized ensemble weights for one label over the planes actually present.

    Renormalizing over `series_types` is what makes missing planes work: a study
    without a fluid coronal simply distributes the coronal model's share across the
    planes that ran.

    Args:
        series_types: The series types of the models that produced predictions for
            this study, in prediction order.
        label: The finding being merged.

    Returns:
        Weights summing to 1.0, aligned with `series_types`.

    Raises:
        KeyError: If a series type has no prior row — non-fluid models need rows
            added to `PLANE_PRIOR` before they can join the ensemble.
        ValueError: If `series_types` is empty.
    """
    if not series_types:
        raise ValueError("combiner_weights needs at least one series type")
    raw = [PLANE_PRIOR[series_type][label] for series_type in series_types]
    total = sum(raw)
    return tuple(grade / total for grade in raw)
