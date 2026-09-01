import math

import pytest

from knee.labels import Label
from knee.plane_prior import PLANE_PRIOR, combiner_weights
from knee.series import SeriesType

_FLUID_TYPES = [SeriesType.SAGITTAL_FLUID, SeriesType.CORONAL_FLUID, SeriesType.AXIAL_FLUID]


def test_prior_covers_every_fluid_plane_and_label() -> None:
    """Catches a new label or ensemble plane silently missing from the prior — the
    combiner would KeyError at scoring time, costing a submission."""
    for series_type in _FLUID_TYPES:
        assert set(PLANE_PRIOR[series_type]) == set(Label)
        assert all(grade >= 1 for grade in PLANE_PRIOR[series_type].values())


def test_weights_normalize_and_favor_the_plane_of_choice() -> None:
    """Catches a transposed or misread matrix — shapes would still look right, but
    e.g. ACL confidence would lean on the axial model instead of the sagittal one."""
    weights = dict(zip(_FLUID_TYPES, combiner_weights(_FLUID_TYPES, Label.ACL)))
    assert math.isclose(sum(weights.values()), 1.0)
    assert weights[SeriesType.SAGITTAL_FLUID] == max(weights.values())

    weights = dict(zip(_FLUID_TYPES, combiner_weights(_FLUID_TYPES, Label.PATELLOFEMORAL_OA)))
    assert weights[SeriesType.AXIAL_FLUID] == max(weights.values())


def test_missing_plane_renormalizes() -> None:
    """Catches weights failing to redistribute when a study lacks a plane — the
    remaining models' shares must sum to 1, not to the leftover fraction."""
    present = [SeriesType.SAGITTAL_FLUID, SeriesType.AXIAL_FLUID]
    weights = combiner_weights(present, Label.MCL)
    assert math.isclose(sum(weights), 1.0)
    # MCL: sagittal is graded limited (1) vs axial useful (2).
    assert weights[1] > weights[0]

    assert combiner_weights([SeriesType.AXIAL_FLUID], Label.ACL) == (1.0,)


def test_empty_plane_list_is_rejected() -> None:
    """Catches the all-models-absent case reaching the weighting math as a division
    by zero instead of being handled by the caller's 0.5 fallback."""
    with pytest.raises(ValueError, match="at least one"):
        combiner_weights([], Label.ACL)
