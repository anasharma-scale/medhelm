"""Deterministic CliniCARE metrics, read from the in-container verifier's reward.json."""

from __future__ import annotations

import json
from typing import Dict, List

from helm.benchmark.adaptation.adapter_spec import AdapterSpec
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.metrics.metric import Metric, MetricMetadata
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.metric_service import MetricService
from helm.benchmark.metrics.statistic import Stat

# reward.json key -> MedHELM metric name. Keys are conditional: most appear only when measured.
OUTCOME_KEYS: Dict[str, str] = {
    "score": "clinicare_score",
    "outcome_label_parsed": "clinicare_verdict_parse_rate",
    "grounding_id_precision": "clinicare_grounding_precision",
    "policy_file_f1": "clinicare_policy_doc_f1",
    "evidence_calls": "clinicare_tool_calls",
    # Trust signals: recorded in stats.json, kept off the leaderboard by the schema.
    "evidence_calls_auth_error": "clinicare_evidence_calls_auth_error",
    "environment_degraded": "clinicare_environment_degraded",
    "report_present": "clinicare_report_present",
}


class CliniCAREOutcomeMetric(Metric):
    """Never emits a Stat for an unmeasured key: a missing key is omitted, not counted as 0.

    A task with no trial emits nothing, so it shows up as a coverage gap rather than a score.
    """

    def evaluate_generation(
        self,
        adapter_spec: AdapterSpec,
        request_state: RequestState,
        metric_service: MetricService,
        eval_cache_path: str,
    ) -> List[Stat]:
        del adapter_spec, metric_service, eval_cache_path
        result = request_state.result
        if result is None or not result.completions or not result.completions[0].text.strip():
            return []
        reward = json.loads(result.completions[0].text)["reward"]
        return [
            Stat(MetricName(name)).add(float(reward[key]))
            for key, name in OUTCOME_KEYS.items()
            if reward.get(key) is not None
        ]

    def get_metadata(self) -> List[MetricMetadata]:
        return [
            MetricMetadata(
                name="clinicare_score",
                display_name="CliniCARE score",
                short_display_name="Score",
                description="Fraction of cases with the correct four-way verdict (in-container verifier).",
                lower_is_better=False,
                group="accuracy",
            ),
        ]
