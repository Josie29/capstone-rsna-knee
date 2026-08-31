from knee.labels import LABEL_COLUMNS, SUBMISSION_COLUMNS, Label

# Copied verbatim from the competition Evaluation tab's submission-format example.
EXPECTED_HEADER = (
    "StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,"
    "PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture"
)


def test_submission_header_matches_the_competition_spec() -> None:
    """Catches the bug where a renamed or reordered label silently produces a submission
    Kaggle rejects - or worse, scores against the wrong column - after a 9-hour run."""
    assert ",".join(SUBMISSION_COLUMNS) == EXPECTED_HEADER


def test_there_are_exactly_twelve_targets() -> None:
    """The metric is a macro average over twelve columns; a thirteenth or an eleventh
    means the model head and the metric have silently diverged."""
    assert len(Label) == 12
    assert len(LABEL_COLUMNS) == 12
