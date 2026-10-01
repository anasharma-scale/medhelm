"""Runs (or replays) CliniCARE-Bench agent episodes for MedHELM.

Live (default): each request runs one case through the CliniCARE checkout's ``scripts/run_case.py``
in CliniCARE's own virtualenv, then reads the trial it produced. MedHELM never imports CliniCARE
code or its harbor pin; the subprocess command and its JSON output are the whole contract.

Replay (``job_dir=`` on the run entry): reads a finished CliniCARE job dir instead. No Docker, MIMIC
or model calls.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from helm.benchmark.scenarios.clinicare_constants import CLINICARE_PROTOCOL, CLINICARE_RUN_CASE
from helm.benchmark.scenarios.clinicare_scenario import resolve_clinicare_root, resolve_tasks_dir
from helm.clients.client import Client
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.request import ErrorFlags, GeneratedOutput, Request, RequestResult
from helm.proxy.retry import NonRetriableException

# run_case.py exits 2 for usage/preflight errors (missing key, unbuilt tasks, gateway not
# allowlisted): the setup is broken, so every remaining case would fail the same way.
RUN_CASE_USAGE_ERROR = 2
# CliniCARE's agent ceiling is 5400 s, plus up to 3x the 360 s harness setup and the verifier.
DEFAULT_CASE_TIMEOUT_SEC = 7200


def load_model_map() -> Dict[str, Any]:
    map_path = Path(__file__).resolve().parents[1] / "benchmark" / "static" / "clinicare_model_map.yaml"
    with map_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def lookup_system(evaluated_model: str) -> Dict[str, Any]:
    """The model-map row for ``evaluated_model``. Exact match only; an unmapped model fails closed."""
    for row in load_model_map().get("models") or []:
        if row.get("model") == evaluated_model:
            return row
    raise NonRetriableException(
        f"No CliniCARE system for model {evaluated_model!r}; add it to benchmark/static/clinicare_model_map.yaml"
    )


def require_docker() -> None:
    """Fail closed if the Docker daemon is not running: every case runs in containers."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=20)
    except FileNotFoundError as exc:
        raise RuntimeError("CliniCARE needs Docker to run cases, but the docker CLI was not found.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("CliniCARE Docker preflight timed out (`docker info`). Is Docker running?") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() or f"exit {result.returncode}"
        raise RuntimeError(f"CliniCARE needs a running Docker daemon. `docker info` failed: {detail}")


def job_name(system: str, task_id: str, effort: Optional[str] = None) -> str:
    """One harbor job per (system, effort, case). Deterministic, so re-running resumes it."""
    return f"medhelm_{system}" + (f"_eff-{effort}" if effort else "") + f"_{task_id}"


def read_trial(trial_dir: Path) -> Dict[str, Any]:
    """Status, rewards and identity of one finished trial (the run_case.py status rules)."""
    result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    reward_path = trial_dir / "verifier" / "reward.json"
    verified = bool((result.get("verifier_result") or {}).get("rewards")) and reward_path.is_file()
    exception = result.get("exception_info") or {}
    report = trial_dir / "verifier" / "report.md"
    agent = (result.get("config") or {}).get("agent") or {}
    return {
        "status": "dead" if not verified else ("agent_error" if exception else "ok"),
        # harbor truncates trial dir names to 32 chars, so the task id comes from the task path.
        "task_id": Path(((result.get("task_id") or {}).get("path")) or "").name,
        "model_name": agent.get("model_name"),
        "agent_version": (result.get("agent_info") or {}).get("version"),
        "exception_type": exception.get("exception_type"),
        "reward": json.loads(reward_path.read_text(encoding="utf-8")) if verified else None,
        "report_chars": len(report.read_text(encoding="utf-8")) if report.is_file() else 0,
    }


def _completion(trial: Dict[str, Any], system: str) -> RequestResult:
    payload = {key: trial[key] for key in ("status", "reward", "report_chars", "agent_version", "exception_type")}
    payload["system"] = system
    return RequestResult(
        success=True,
        cached=False,
        completions=[GeneratedOutput(text=json.dumps(payload), logprob=0.0, tokens=[])],
        embedding=[],
    )


def _failed(error: str, fatal: bool) -> RequestResult:
    # Non-fatal: MedHELM records an empty completion and keeps going. Never retriable: a dead trial
    # or a broken setup will not fix itself across HELM's 5 backoff retries.
    return RequestResult(
        success=False,
        cached=False,
        completions=[],
        embedding=[],
        error=error,
        error_flags=ErrorFlags(is_retriable=False, is_fatal=fatal),
    )


class CliniCAREClient(Client):
    """Serves each task's trial as the completion: JSON ``{"status", "reward", "report_chars", ...}``.

    Refuses (NonRetriableException) an unmapped model, a trial produced by a different model than
    the map declares, a replay job dir that mixes models, or one holding tasks absent from the tasks
    dir. Does not cache: the harbor job dir is the cache (a finished case is never re-run).
    """

    def __init__(self, cache_config: Optional[CacheConfig] = None, **kwargs: Any):
        del cache_config, kwargs
        self._lock = threading.Lock()
        self._replay_index: Dict[str, Dict[str, Path]] = {}
        self._docker_checked = False

    def make_request(self, request: Request) -> RequestResult:
        envelope = json.loads(request.prompt)
        if envelope.get("clinicare_protocol") != CLINICARE_PROTOCOL:
            raise NonRetriableException(f"Expected protocol {CLINICARE_PROTOCOL}, got {envelope!r:.200}")
        row = lookup_system(envelope["evaluated_model"])
        if envelope.get("job_dir"):
            return self._replay(envelope, row)
        return self._live(envelope, row)

    # -- live -------------------------------------------------------------------------------------

    def _live(self, envelope: Dict[str, Any], row: Dict[str, Any]) -> RequestResult:
        root = resolve_clinicare_root(envelope.get("clinicare_root", ""))
        if root is None:
            raise NonRetriableException(
                "CliniCARE live mode needs the CliniCARE checkout: set clinicare_root= or export CLINICARE_ROOT"
            )
        python = root / ".venv" / "bin" / "python"
        if not (root / CLINICARE_RUN_CASE).is_file():
            raise NonRetriableException(f"{root} has no {CLINICARE_RUN_CASE}; update the CliniCARE checkout")
        if not python.is_file():
            raise NonRetriableException(f"No CliniCARE virtualenv at {python}; run `uv sync` in {root}")
        with self._lock:
            if not self._docker_checked:
                try:
                    require_docker()
                except RuntimeError as exc:
                    return _failed(str(exc), fatal=True)
                self._docker_checked = True

        task_id = envelope["task_id"]
        system = row["system"]
        effort = envelope.get("effort") or row.get("effort")
        tasks_dir = resolve_tasks_dir(envelope.get("tasks_dir", ""), str(root))
        jobs_dir = Path(envelope.get("jobs_dir") or root / "jobs" / "medhelm").expanduser()
        cmd = [
            str(python),
            CLINICARE_RUN_CASE,
            "--system",
            system,
            "--task-id",
            task_id,
            "--tasks",
            str(tasks_dir),
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            job_name(system, task_id, effort),
        ]
        if effort:
            cmd += ["--effort", effort]
        if envelope.get("grade"):
            cmd += ["--grade"]
        timeout = int(envelope.get("case_timeout_sec") or DEFAULT_CASE_TIMEOUT_SEC)
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout, env=os.environ.copy())
        except subprocess.TimeoutExpired:
            return _failed(f"CliniCARE case timed out after {timeout}s", fatal=False)
        if proc.returncode == RUN_CASE_USAGE_ERROR:
            # Broken setup (missing key, unbuilt tasks, gateway not allowlisted): abort the run
            # rather than record hundreds of empty results.
            return _failed(f"CliniCARE run_case refused: {proc.stderr.strip()[-2000:]}", fatal=True)
        if proc.returncode != 0:
            return _failed(f"CliniCARE run_case exited {proc.returncode}: {proc.stderr.strip()[-2000:]}", fatal=False)
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        trial = read_trial(Path(report["trial_dir"]))
        self._check_attribution(trial["model_name"], row, Path(report["trial_dir"]))
        if trial["status"] == "dead":
            return _failed(f"CliniCARE trial died ({trial['exception_type']}); see {report['trial_dir']}", fatal=False)
        return _completion(trial, system)

    # -- replay -----------------------------------------------------------------------------------

    def _index_job(self, job_dir: str, tasks_dir: Path, row: Dict[str, Any]) -> Dict[str, Path]:
        job = Path(job_dir).expanduser()
        trials: Dict[str, Path] = {}
        models = set()
        for trial in sorted(job.glob("*__*/")):
            info = read_trial(trial)
            if info["task_id"] in trials:
                raise NonRetriableException(f"Job dir {job} has more than one trial for one task")
            trials[info["task_id"]] = trial
            models.add(info["model_name"])
        if len(models) > 1:
            raise NonRetriableException(f"Job dir {job} mixes models {sorted(models)}; one job dir per model")
        for model in models:
            self._check_attribution(model, row, job)
        foreign = [task_id for task_id in trials if not (tasks_dir / task_id).is_dir()]
        if foreign:
            raise NonRetriableException(
                f"{len(foreign)} trial(s) in {job} have no task in {tasks_dir}; the two are from different builds"
            )
        n_tasks = sum(1 for d in tasks_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")))
        hlog(f"CliniCARE replay: {len(trials)} trials for {n_tasks} tasks ({job.name})")
        return trials

    def _replay(self, envelope: Dict[str, Any], row: Dict[str, Any]) -> RequestResult:
        key = envelope["job_dir"]
        tasks_dir = resolve_tasks_dir(envelope.get("tasks_dir", ""), envelope.get("clinicare_root", ""))
        with self._lock:
            if key not in self._replay_index:
                self._replay_index[key] = self._index_job(key, tasks_dir, row)
        trial_dir = self._replay_index[key].get(envelope["task_id"])
        if trial_dir is None:
            return _failed(f"no finished trial for task {envelope['task_id']}", fatal=False)
        trial = read_trial(trial_dir)
        if trial["status"] == "dead":
            return _failed(f"CliniCARE trial died ({trial['exception_type']})", fatal=False)
        return _completion(trial, row["system"])

    @staticmethod
    def _check_attribution(model_name: Optional[str], row: Dict[str, Any], where: Path) -> None:
        if model_name != row["harbor_model"]:
            raise NonRetriableException(
                f"{where} was produced by {model_name!r}, but the model map says {row['model']!r} runs as "
                f"{row['harbor_model']!r}; refusing to attribute these results"
            )
