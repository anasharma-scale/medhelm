"""Reads a finished CliniCARE (Harbor) job dir. Runs nothing: no Docker, MIMIC or model calls."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from helm.benchmark.scenarios.clinicare_constants import CLINICARE_PROTOCOL
from helm.clients.client import Client
from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.request import ErrorFlags, GeneratedOutput, Request, RequestResult
from helm.proxy.retry import NonRetriableException


class CliniCAREReplayClient(Client):
    """Serves each task's finished trial as the completion.

    The completion is JSON ``{"reward": <verifier/reward.json>, "report_chars": int}``, so metrics
    never touch the job dir. On first use of a job dir it refuses (NonRetriableException) when the
    job mixes models, was produced by a different model than the run entry declares, or contains
    tasks absent from the tasks dir (mismatched builds).
    """

    def __init__(self, cache_config: Optional[CacheConfig] = None, **kwargs: Any):
        del cache_config, kwargs
        self._lock = threading.Lock()
        self._trials: Dict[str, Dict[str, Path]] = {}

    def _index_job(self, job_dir: str, tasks_dir: str, evaluated_model: str) -> Dict[str, Path]:
        job, tasks = Path(job_dir).expanduser(), Path(tasks_dir).expanduser()
        trials: Dict[str, Path] = {}
        models = set()
        for trial in sorted(job.glob("*__*/")):
            result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
            task_id = trial.name.split("__")[0]  # trial dirs are <task_id>__<suffix>
            if task_id in trials:
                raise NonRetriableException(f"Job dir {job} has more than one trial for one task")
            trials[task_id] = trial
            models.add(result["config"]["agent"]["model_name"])
        if len(models) > 1:
            raise NonRetriableException(f"Job dir {job} mixes models {sorted(models)}; one job dir per model")
        if models and models != {evaluated_model}:
            raise NonRetriableException(
                f"Job dir {job} was produced by {sorted(models)[0]!r} but the run entry declares "
                f"{evaluated_model!r}; refusing to attribute these results"
            )
        foreign = [task_id for task_id in trials if not (tasks / task_id).is_dir()]
        if foreign:
            raise NonRetriableException(
                f"{len(foreign)} trial(s) in {job} have no task in {tasks}; the two are from different builds"
            )
        n_tasks = sum(1 for d in tasks.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")))
        hlog(f"CliniCARE replay: {len(trials)} trials for {n_tasks} tasks ({job.name})")
        return trials

    def make_request(self, request: Request) -> RequestResult:
        envelope = json.loads(request.prompt)
        if envelope.get("clinicare_protocol") != CLINICARE_PROTOCOL:
            raise NonRetriableException(f"Expected protocol {CLINICARE_PROTOCOL}, got {envelope!r:.200}")
        key = envelope["job_dir"]
        with self._lock:
            if key not in self._trials:
                self._trials[key] = self._index_job(key, envelope["tasks_dir"], envelope["evaluated_model"])
        task_id = envelope["task_id"]
        trial = self._trials[key].get(task_id)
        reward_path = trial / "verifier" / "reward.json" if trial else None
        if reward_path is None or not reward_path.is_file():
            # Missing trial: non-fatal so the run continues, non-retriable since it will never appear.
            return RequestResult(
                success=False,
                cached=False,
                completions=[],
                embedding=[],
                error=f"no finished trial for task {task_id}",
                error_flags=ErrorFlags(is_retriable=False, is_fatal=False),
            )
        report = trial / "verifier" / "report.md" if trial else None
        payload = {
            "reward": json.loads(reward_path.read_text(encoding="utf-8")),
            "report_chars": len(report.read_text(encoding="utf-8")) if report and report.is_file() else 0,
        }
        return RequestResult(
            success=True,
            cached=False,
            completions=[GeneratedOutput(text=json.dumps(payload), logprob=0.0, tokens=[])],
            embedding=[],
        )
