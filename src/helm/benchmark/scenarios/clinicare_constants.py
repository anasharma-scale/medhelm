"""Shared constants for the CliniCARE-Bench MedHELM integration."""

# Where the CliniCARE checkout lives (live mode runs cases through it) and, optionally, which built
# tasks dir to use (default: <CLINICARE_ROOT>/benchmark/tasks).
CLINICARE_ROOT_ENV = "CLINICARE_ROOT"
CLINICARE_TASKS_DIR_ENV = "CLINICARE_TASKS_DIR"
# The CliniCARE entry point that runs one case and prints the trial as JSON.
CLINICARE_RUN_CASE = "scripts/run_case.py"

# Gold verdicts as they appear in the cohort bundle's expected_label column.
VERDICT_LABELS = (
    "YES",
    "NO",
    "INDETERMINATE_LACK_OF_DATA",
    "INDETERMINATE_MEDICAL_AMBIGUITY",
)

CLINICARE_PROTOCOL = "clinicare.v1"
# Internal routing deployment. Leaderboard rows use the evaluated model, never this name.
CLINICARE_HARNESS_DEPLOYMENT = "clinicare/harness"
