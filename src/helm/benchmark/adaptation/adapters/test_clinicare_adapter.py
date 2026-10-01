import json
from dataclasses import replace

from helm.benchmark.adaptation.adapters.clinicare_adapter import build_clinicare_request
from helm.benchmark.run_specs.medhelm_run_specs import get_clinicare_spec
from helm.benchmark.scenarios.clinicare_constants import CLINICARE_HARNESS_DEPLOYMENT, CLINICARE_PROTOCOL
from helm.benchmark.scenarios.scenario import TEST_SPLIT, Input, Instance

TASK = "advanced-imaging-000001"  # placeholder: real task ids identify patients


def _instance() -> Instance:
    return Instance(
        input=Input(text="placeholder prompt"),
        references=[],
        split=TEST_SPLIT,
        id=TASK,
        extra_data={"task_id": TASK, "scenario_id": "advanced-imaging"},
    )


def test_request_keeps_evaluated_model_and_reroutes_deployment():
    spec = get_clinicare_spec(clinicare_root="/opt/clinicare", effort="high", grade="true")
    adapter_spec = replace(spec.adapter_spec, model="openai/gpt-5.6-sol", model_deployment="openai/gpt-5.6-sol")
    request = build_clinicare_request(_instance(), adapter_spec)

    assert request.model == "openai/gpt-5.6-sol"
    assert request.model_deployment == CLINICARE_HARNESS_DEPLOYMENT
    envelope = json.loads(request.prompt)
    assert envelope["clinicare_protocol"] == CLINICARE_PROTOCOL
    assert envelope["task_id"] == TASK
    assert envelope["evaluated_model"] == "openai/gpt-5.6-sol"
    assert envelope["clinicare_root"] == "/opt/clinicare"
    assert envelope["effort"] == "high"
    assert envelope["grade"] is True
    assert "job_dir" not in envelope  # empty knobs are dropped, so live mode is the default


def test_replay_knob_reaches_envelope():
    spec = get_clinicare_spec(tasks_dir="/t", job_dir="/j")
    request = build_clinicare_request(_instance(), replace(spec.adapter_spec, model="openai/gpt-5.6-sol"))
    envelope = json.loads(request.prompt)
    assert envelope["job_dir"] == "/j" and envelope["tasks_dir"] == "/t"
    assert "grade" not in envelope


def test_run_spec_has_outcome_and_offline_metrics():
    classes = [m.class_name for m in get_clinicare_spec().metric_specs]
    assert "helm.benchmark.metrics.clinicare_metrics.CliniCAREOutcomeMetric" in classes
    assert "helm.benchmark.metrics.clinicare_metrics.CliniCAREOfflineMetric" in classes
