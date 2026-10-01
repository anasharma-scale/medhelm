"""Shared constants for the CliniCARE-Bench MedHELM integration."""

CLINICARE_TASKS_DIR_ENV = "CLINICARE_TASKS_DIR"

# Gold verdicts as they appear in the cohort bundle's expected_label column.
VERDICT_LABELS = (
    "YES",
    "NO",
    "INDETERMINATE_LACK_OF_DATA",
    "INDETERMINATE_MEDICAL_AMBIGUITY",
)

CLINICARE_PROTOCOL = "clinicare.v1"
# Internal routing deployment. Leaderboard rows use the evaluated model, never this name.
CLINICARE_REPLAY_DEPLOYMENT = "clinicare/replay"
