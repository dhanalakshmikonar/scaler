from __future__ import annotations

from .models import EnvironmentState, RewardBreakdown


def grade_final_submission(state: EnvironmentState) -> RewardBreakdown:
    scenario = state.scenario
    investigation_hits = len(state.discovered_evidence_ids)
    investigation_reward = min(0.45, investigation_hits * 0.12)
    efficiency_penalty = max(-0.2, -0.03 * max(0, state.steps_taken - 3))
    verdict_reward = 0.0
    confidence_adjustment = 0.0

    if state.submitted_verdict is not None:
        if state.submitted_verdict == scenario.verdict:
            verdict_reward = 0.45
            confidence = state.submitted_confidence or 0.0
            confidence_adjustment = 0.1 if confidence >= 0.6 else 0.03
        else:
            verdict_reward = -0.4
            confidence = state.submitted_confidence or 0.0
            confidence_adjustment = -0.1 if confidence >= 0.7 else -0.03

    total = investigation_reward + efficiency_penalty + verdict_reward + confidence_adjustment
    return RewardBreakdown(
        investigation_reward=investigation_reward,
        efficiency_penalty=efficiency_penalty,
        verdict_reward=verdict_reward,
        confidence_adjustment=confidence_adjustment,
        total=max(-1.0, min(1.0, total)),
    )
