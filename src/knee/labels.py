from enum import StrEnum


class Label(StrEnum):
    """The twelve target findings, named exactly as the submission columns require.

    Column names carry spaces and an apostrophe; a submission whose header does not match
    character-for-character is rejected, so these strings are the single source of truth.
    """

    ACL = "ACL"
    MCL = "MCL"
    MEDIAL_MENISCUS = "Medial Meniscus"
    LATERAL_MENISCUS = "Lateral Meniscus"
    MEDIAL_OA = "Medial OA"
    LATERAL_OA = "Lateral OA"
    PATELLOFEMORAL_OA = "PF OA"
    EFFUSION = "Effusion"
    SYNOVITIS = "Synovitis"
    BAKERS_CYST = "Baker's"
    CONTUSION = "Contusion"
    FRACTURE = "Fracture"


STUDY_ID_COLUMN = "StudyInstanceUID"

# Order is load-bearing: submission.csv columns must appear in this sequence.
LABEL_COLUMNS: tuple[str, ...] = tuple(label.value for label in Label)

SUBMISSION_COLUMNS: tuple[str, ...] = (STUDY_ID_COLUMN, *LABEL_COLUMNS)


class AnatomicalPlane(StrEnum):
    """Values of the `Anatomical_Plane` column in train_series.csv / test_series.csv."""

    SAGITTAL = "Sagittal"
    CORONAL = "Coronal"
    AXIAL = "Axial"
