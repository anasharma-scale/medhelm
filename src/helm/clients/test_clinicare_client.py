import json
import os
import subprocess
import sys
import time
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


def _trial(
    job: Path,
    task_id: str,
    *,
    model: str = MODEL,
    verified: bool = True,
    exc: Optional[str] = None,
    reward: Optional[Dict[str, Any]] = None,
) -> Path:
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
        (trial / "verifier" / "reward.json").write_text(json.dumps(reward or {"score": 1.0, "evidence_calls": 3}))
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
def _fresh_client_state(monkeypatch):
    # CliniCAREClient keeps run-wide state on the class (shared across the instances AutoClient may
    # build); give every test a clean run.
    monkeypatch.setattr(CliniCAREClient, "_replay_index", {})
    monkeypatch.setattr(CliniCAREClient, "_docker_checked", False)
    monkeypatch.setattr(CliniCAREClient, "_warned_ungraded", False)


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
    monkeypatch,
    *,
    returncode: int = 0,
    trial: Optional[Path] = None,
    stderr: str = "",
    graded: Optional[bool] = None,
) -> List[List[str]]:
    calls: List[List[str]] = []

    def fake_run(cmd, timeout, **kwargs):
        calls.append(list(cmd))
        report: Dict[str, Any] = {"status": "ok", "trial_dir": str(trial)}
        if graded is not None:
            report["graded"] = graded
        stdout = json.dumps(report) + "\n" if trial else ""
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(clinicare_client, "require_docker", lambda: None)
    monkeypatch.setattr(clinicare_client, "run_in_process_group", fake_run)
    return calls


def _payload(result) -> Dict[str, Any]:
    return json.loads(result.completions[0].text)


def test_unmapped_model_fails_closed():
    envelope = {"clinicare_protocol": CLINICARE_PROTOCOL, "task_id": TASK, "evaluated_model": "openai/gpt-4o"}
    with pytest.raises(NonRetriableException, match="No CliniCARE system"):
        CliniCAREClient().make_request(Request(model="openai/gpt-4o", prompt=json.dumps(envelope)))


def test_job_name_is_deterministic_and_separates_builds(tmp_path):
    smoke, full = tmp_path / "tasks_smoke", tmp_path / "tasks"
    name = job_name("codex-gpt56sol", TASK, full)
    assert name == job_name("codex-gpt56sol", TASK, full)  # re-run resumes the same job
    assert name.startswith("medhelm_codex-gpt56sol_") and name.endswith(f"_{TASK}")
    # Same task id in two builds must not share a job dir (harbor refuses to resume across them).
    assert name != job_name("codex-gpt56sol", TASK, smoke)
    assert "_eff-high_" in job_name("codex-gpt56sol", TASK, full, "high")


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
    assert cmd[cmd.index("--job-name") + 1] == job_name("codex-gpt56sol", TASK, live_root / "benchmark" / "tasks")
    assert "--grade" not in cmd


def test_live_passes_grade_and_effort(monkeypatch, live_root, tmp_path):
    calls = _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK))
    CliniCAREClient().make_request(_request(clinicare_root=str(live_root), grade=True, effort="high"))
    assert "--grade" in calls[0]
    assert calls[0][calls[0].index("--effort") + 1] == "high"


@pytest.mark.parametrize(
    "returncode,stderr",
    [
        (2, "run_case: OPENAI_API_KEY not set"),  # preflight refusal
        (2, "run_case: harbor refused the job (exit 1): FileExistsError"),  # stale job dir
        (1, "Traceback (most recent call last): ModuleNotFoundError"),  # run_case itself crashed
    ],
)
def test_any_run_case_failure_is_fatal(monkeypatch, live_root, returncode, stderr):
    # Per-case outcomes always come back as exit 0, so any other exit is a broken setup: abort the
    # run instead of recording an empty result for every remaining case.
    _fake_run_case(monkeypatch, returncode=returncode, stderr=stderr)
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert not result.success
    assert result.error_flags is not None and result.error_flags.is_fatal and not result.error_flags.is_retriable


def test_timeout_kills_the_whole_process_group(tmp_path):
    # run_case.py starts harbor as a child; killing only run_case.py would leave harbor running.
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        clinicare_client.run_in_process_group([sys.executable, "-c", script], 2, stdout=subprocess.PIPE)
    grandchild = int(pid_file.read_text())
    for _ in range(50):  # the kill is asynchronous; give the OS a moment to reap it
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(grandchild, 9)
    pytest.fail("grandchild survived the timeout")


def test_live_dead_trial_is_non_fatal_and_not_retried(monkeypatch, live_root, tmp_path):
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK, verified=False, exc="RuntimeError"))
    result = CliniCAREClient().make_request(_request(clinicare_root=str(live_root)))
    assert not result.success
    assert result.error_flags is not None
    assert not result.error_flags.is_fatal and not result.error_flags.is_retriable


def test_live_timeout_is_non_fatal(monkeypatch, live_root):
    def timeout(cmd, timeout, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(clinicare_client, "require_docker", lambda: None)
    monkeypatch.setattr(clinicare_client, "run_in_process_group", timeout)
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


@pytest.mark.parametrize("make_dir", [False, True])
def test_replay_missing_or_empty_job_dir_raises(tmp_path, make_dir):
    # A mistyped job_dir must not produce a "successful" run with no scores.
    tasks = _tasks(tmp_path, TASK)
    job = tmp_path / "job"
    if make_dir:
        job.mkdir()
    with pytest.raises(NonRetriableException, match="not found|no finished trials"):
        CliniCAREClient().make_request(_request(job_dir=str(job), tasks_dir=str(tasks)))


def test_replay_foreign_task_raises(tmp_path):
    tasks = _tasks(tmp_path, TASK)
    _trial(tmp_path / "job", TASK)
    _trial(tmp_path / "job", OTHER)
    with pytest.raises(NonRetriableException, match="different builds"):
        CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))


@pytest.mark.parametrize(
    "reward,status",
    [
        # The verifier failed before scoring; its 0.0 is not a verdict (judge.py / test.sh paths).
        ({"score": 0.0, "judge_failed": 1, "rubrics_load_failed": 1}, "dead"),
        ({"score": 0.0, "judge_failed": 1, "judge_launch_failed": 1}, "dead"),
        # A missing report is a genuine 0, and judge_failed alone flags only the process pass.
        ({"score": 0.0, "report_present": 0, "report_chars": 0}, "ok"),
        ({"score": 1.0, "outcome_score": 1.0, "judge_failed": 1}, "ok"),
    ],
)
def test_verifier_failure_is_unscored_not_a_wrong_verdict(tmp_path, reward, status):
    trial = read_trial(_trial(tmp_path / "job", TASK, reward=reward))
    assert trial["status"] == status
    assert (trial["reward"] is None) == (status == "dead")


def test_replay_verifier_failure_is_a_coverage_gap(tmp_path):
    tasks = _tasks(tmp_path, TASK)
    _trial(tmp_path / "job", TASK, reward={"score": 0.0, "judge_failed": 1, "rubrics_load_failed": 1})
    result = CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks)))
    assert not result.success  # -> clinicare_trial_failed, not clinicare_score = 0
    assert result.error_flags is not None and not result.error_flags.is_fatal


GRADED_REWARD = {"score": 1.0, "process_pct": 75.0, "policy_support_judged": 1, "policy_citation_support_rate": 0.5}


def _warnings(monkeypatch) -> List[str]:
    seen: List[str] = []
    monkeypatch.setattr(clinicare_client, "hwarn", lambda msg, **kw: seen.append(msg))
    return seen


def test_live_grade_requested_but_not_graded_warns_once(monkeypatch, live_root, tmp_path):
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK), graded=False)
    warnings = _warnings(monkeypatch)
    client = CliniCAREClient()
    for _ in range(3):  # three ungraded cases in one run -> one warning, three counted in the stat
        payload = _payload(client.make_request(_request(clinicare_root=str(live_root), grade=True)))
        assert (payload["grade_requested"], payload["graded"]) == (True, False)
        assert payload["reward"]["score"] == 1.0  # the deterministic score is kept
    assert len(warnings) == 1 and "JUDGE_BASE_URL" in warnings[0]


def test_live_graded_case_does_not_warn(monkeypatch, live_root, tmp_path):
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK, reward=GRADED_REWARD), graded=True)
    warnings = _warnings(monkeypatch)
    payload = _payload(CliniCAREClient().make_request(_request(clinicare_root=str(live_root), grade=True)))
    assert (payload["grade_requested"], payload["graded"]) == (True, True)
    assert warnings == []


def test_live_without_grade_never_warns(monkeypatch, live_root, tmp_path):
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK), graded=False)
    warnings = _warnings(monkeypatch)
    payload = _payload(CliniCAREClient().make_request(_request(clinicare_root=str(live_root))))
    assert payload["grade_requested"] is False
    assert warnings == []


def test_live_older_run_case_without_graded_flag_falls_back_to_reward(monkeypatch, live_root, tmp_path):
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK, reward=GRADED_REWARD))  # no flag
    payload = _payload(CliniCAREClient().make_request(_request(clinicare_root=str(live_root), grade=True)))
    assert payload["graded"] is True


def test_live_previously_graded_case_counts_as_graded(monkeypatch, live_root, tmp_path):
    # This run's judges failed (e.g. keys now missing), but an earlier run already graded the case:
    # the judged columns are filled, so no warning.
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK, reward=GRADED_REWARD), graded=False)
    warnings = _warnings(monkeypatch)
    payload = _payload(CliniCAREClient().make_request(_request(clinicare_root=str(live_root), grade=True)))
    assert payload["graded"] is True and warnings == []


@pytest.mark.parametrize("reward,graded", [({"score": 1.0}, False), (GRADED_REWARD, True)])
def test_replay_grade_requested_checks_the_reward(monkeypatch, tmp_path, reward, graded):
    tasks = _tasks(tmp_path, TASK)
    _trial(tmp_path / "job", TASK, reward=reward)
    warnings = _warnings(monkeypatch)
    result = CliniCAREClient().make_request(_request(job_dir=str(tmp_path / "job"), tasks_dir=str(tasks), grade=True))
    assert _payload(result)["graded"] is graded
    assert len(warnings) == (0 if graded else 1)
    if not graded:
        assert "Replay only reads results" in warnings[0]


def test_warning_and_docker_check_are_shared_across_client_instances(monkeypatch, live_root, tmp_path):
    # With --num-threads N, HELM's AutoClient can build N instances of this client concurrently. A
    # real 5-thread replay logged the "not graded" warning 5 times before this was shared.
    _fake_run_case(monkeypatch, trial=_trial(tmp_path / "jobs" / "j", TASK), graded=False)
    docker_checks: List[int] = []
    monkeypatch.setattr(clinicare_client, "require_docker", lambda: docker_checks.append(1))
    warnings = _warnings(monkeypatch)
    for _ in range(5):
        CliniCAREClient().make_request(_request(clinicare_root=str(live_root), grade=True))
    assert len(warnings) == 1
    assert len(docker_checks) == 1
