import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from helm.benchmark.scenarios.clinicare_constants import CLINICARE_PROTOCOL, CLINICARE_RUN_CASE
from helm.clients import clinicare_client
from helm.clients.clinicare_client import CliniCAREClient, job_name, read_trial
from helm.common.request import Request
from helm.proxy.retry import NonRetriableException

# Placeholder ids only: real CliniCARE task ids identify patients.
TASK = "advanced-imaging-000001"
OTHER = "sepsis-bundle-000002"
MODEL = "openai/gpt-5.6-sol"
ROW = {"model": MODEL, "system": "codex-gpt56sol", "harbor_model": MODEL, "effort": None}


def _trial(job: Path, task_id: str, *, model: str = MODEL, verified: bool = True, exc: Optional[str] = None) -> Path:
    trial = job / f"{task_id[:32]}__abcdefg"
    (trial / "verifier").mkdir(parents=True)
    result = {
        "task_id": {"path": f"/x/benchmark/tasks/{task_id}"},
        "config": {"agent": {"name": "pkg:Agent", "model_name": model}},
        "agent_info": {"version": "1.0"},
        "verifier_result": {"rewards": {"reward": 1.0}} if verified else None,
        "exception_info": {"exception_type": exc} if exc else None,
    }
    (trial / "result.json").write_text(json.dumps(result))
    if verified:
        (trial / "verifier" / "reward.json").write_text(json.dumps({"score": 1.0, "evidence_calls": 3}))
        (trial / "verifier" / "report.md").write_text("placeholder report")
    return trial


def _tasks(root: Path, *task_ids: str) -> Path:
    tasks = root / "benchmark" / "tasks"
    for task_id in task_ids:
        (tasks / task_id).mkdir(parents=True)
    return tasks


def _request(**knobs: Any) -> Request:
    envelope = {"clinicare_protocol": CLINICARE_PROTOCOL, "task_id": TASK, "evaluated_model": MODEL, **knobs}
    return Request(model=MODEL, model_deployment="clinicare/harness", prompt=json.dumps(envelope))


@pytest.fixture(autouse=True)
def _one_row_map(monkeypatch):
    monkeypatch.setattr(clinicare_client, "load_model_map", lambda: {"models": [dict(ROW)]})


@pytest.fixture
def live_root(tmp_path: Path) -> Path:
    root = tmp_path / "clinicare"
    (root / "scripts").mkdir(parents=True)
    (root / CLINICARE_RUN_CASE).write_text("")
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    _tasks(root, TASK)
    return root


def _fake_run_case(
    monkeypatch, *, returncode: int = 0, trial: Optional[Path] = None, stderr: str = ""
) -> List[List[str]]:
    calls: List[List[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        stdout = json.dumps({"status": "ok", "trial_dir": str(trial)}) + "\n" if trial else ""
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(clinicare_client, "require_docker", lambda: None)
    monkeypatch.setattr(clinicare_client.subprocess, "run", fake_run)
    return calls


def _payload(result) -> Dict[str, Any]:
    return json.loads(result.completions[0].text)


def test_unmapped_model_fails_closed():
    envelope = {"clinicare_protocol": CLINICARE_PROTOCOL, "task_id": TASK, "evaluated_model": "openai/gpt-4o"}
    with pytest.raises(NonRetriableException, match="No CliniCARE system"):
        CliniCAREClient().make_request(Request(model="openai/gpt-4o", prompt=json.dumps(envelope)))


def test_job_name_is_deterministic():
    assert job_name("codex-gpt56sol", TASK) == f"medhelm_codex-gpt56sol_{TASK}"
    assert job_name("codex-gpt56sol", TASK, "high") == f"medhelm_codex-gpt56sol_eff-high_{TASK}"


def test_read_trial_takes_task_id_from_task_path(tmp_path):
    long_id = "placeholder-scenario-with-a-long-name-000001"
    trial = _trial(tmp_path / "job", long_id)
    assert not trial.name.startswith(long_id)
    assert read_trial(trial)["task_id"] == long_id


@pytest.mark.parametrize(
    "verified,exc,status",
    [(True, None, "ok"), (True, "AgentTimeoutError", "agent_error"), (False, "RuntimeError", "dead")],
)
def test_read_trial_status(tmp_path, verified, exc, status):
    assert read_trial(_trial(tmp_path / "job", TASK, verified=verified, exc=exc))["status"] == status


def test_live_runs_run_case_and_returns_reward(monkeypatch, live_root, tmp_path):
    trial = _trial(tmp_path / "jobs" / "j", TASK)
    calls = _fake_run_case(monkeypatch, trial=trial)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))

    assert result.success
    assert _payload(result)["reward"]["score"] == 1.0
    cmd = calls[0]
    assert cmd[:2] == [str(live_root / ".venv" / "bin" / "python"), CLINICARE_RUN_CASE]
    assert cmd[cmd.index("--system") + 1] == "codex-gpt56sol"
    assert cmd[cmd.index("--task-id") + 1] == TASK
    assert cmd[cmd.index("--job-name") + 1] == job_name("codex-gpt56sol", TASK)
    assert "--grade" not in cmd


def test_live_passes_grade_and_effort(monkeypatch, live_root, tmp_path):
    calls = _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK))
    CliniCAREClient().make_request(_request(clinicare_root=str(live_root), grade=True, effort="high"))
    assert "--grade" in calls[0]
    assert calls[0][calls[0].index("--effort") + 1] == "high"


def test_live_preflight_refusal_is_fatal(monkeypatch, live_root):
    _fake_run_case(monkeypatch, returncode=2, stderr="OPENAI_API_KEY not set")
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert not result.success
    assert result.error_flags is not None and result.error_flags.is_fatal and not result.error_flags.is_retriable


def test_live_dead_trial_is_non_fatal_and_not_retried(monkeypatch, live_root, tmp_path):
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK, verified=False, exc="RuntimeError"))
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert not result.success
    assert result.error_flags is not None
    assert not result.error_flags.is_fatal and not result.error_flags.is_retriable


def test_live_timeout_is_non_fatal(monkeypatch, live_root):
    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(clinicare_client, "require_docker", lambda: None)
    monkeypatch.setattr(clinicare_client.subprocess, "run", timeout)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert result.error_flags is not None and not result.error_flags.is_fatal


def test_live_misattributed_trial_raises(monkeypatch, live_root, tmp_path):
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK, model="openai/gpt-5.4"))
    with pytest.raises(NonRetriableException, match="refusing to attribute"):
        CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))


def test_live_missing_docker_is_fatal(monkeypatch, live_root):
    def no_docker():
        raise RuntimeError("no docker")

    monkeypatch.setattr(clinicare_client, "require_docker", no_docker)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert result.error_flags is not None and result.error_flags.is_fatal


def test_live_old_checkout_raises(tmp_path):
    (tmp_path / "old").mkdir()
    with pytest.raises(NonRetriableException, match="update the CliniCARE checkout"):
        CliniCAREClient().make_request(_request(clinicare_root=str(tmp_path / "old")))


def test_replay_serves_trial(tmp_path):
    tasks = _tasks(tmp_path, TASK, OTHER)
    _trial(tmp_path / "job", TASK)
    result = CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))
    assert result.success and _payload(result)["status"] == "ok"


def test_replay_missing_trial_is_non_fatal(tmp_path):
    tasks = _tasks(tmp_path, TASK, OTHER)
    _trial(tmp_path / "job", OTHER)
    result = CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))
    assert result.error_flags is not None and not result.error_flags.is_fatal and not result.error_flags.is_retriable


def test_replay_mixed_models_raise(tmp_path):
    tasks = _tasks(tmp_path, TASK, OTHER)
    _trial(tmp_path / "job", TASK)
    _trial(tmp_path / "job", OTHER, model="openai/gpt-5.4")
    with pytest.raises(NonRetriableException, match="mixes models"):
        CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))


def test_replay_misattributed_job_raises(tmp_path):
    tasks = _tasks(tmp_path, TASK)
    _trial(tmp_path / "job", TASK, model="openai/gpt-5.4")
    with pytest.raises(NonRetriableException, match="refusing to attribute"):
        CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))


def test_replay_foreign_task_raises(tmp_path):
    tasks = _tasks(tmp_path, TASK)
    _trial(tmp_path / "job", TASK)
    _trial(tmp_path / "job", OTHER)
    with pytest.raises(NonRetriableException, match="different builds"):
        CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))
