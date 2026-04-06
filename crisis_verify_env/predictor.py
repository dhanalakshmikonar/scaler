from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import Difficulty, Scenario, Verdict
from .tasks import get_scenarios


@dataclass
class PredictionResult:
    verdict: Verdict
    confidence: float
    risk_level: str
    matched_scenario_id: str
    difficulty_hint: Difficulty
    explanation: str
    signals: list[str]
    suggested_checks: list[str]
    scope_status: str


KEYWORD_RULES: dict[Verdict, set[str]] = {
    Verdict.FAKE: {
        "deepfake",
        "surrender",
        "fake bulletin",
        "spoof",
        "curfew",
        "hoax",
        "forged",
        "leaked video",
    },
    Verdict.MISLEADING: {
        "viral image",
        "completely destroyed",
        "blackout",
        "reused",
        "out of context",
        "satellite",
        "bridge",
        "hospital",
        "protest",
        "warehouse",
    },
    Verdict.REAL: {
        "official",
        "evacuation",
        "corridor",
        "government notice",
        "verified",
        "emergency office",
    },
    Verdict.INSUFFICIENT: {
        "rumor",
        "unconfirmed",
        "reports say",
        "maybe",
        "possibly",
        "unclear",
    },
}

CRISIS_SCOPE_TERMS = {
    "war",
    "crisis",
    "strike",
    "missile",
    "shelling",
    "troop",
    "border",
    "blackout",
    "evacuation",
    "corridor",
    "hospital",
    "camp",
    "aid",
    "ceasefire",
    "refugee",
    "military",
    "convoy",
    "attack",
    "conflict",
    "general",
    "surrender",
    "protest",
    "curfew",
    "cyberattack",
}


def _token_score(text: str, keywords: Iterable[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _scenario_match_score(text: str, scenario: Scenario) -> int:
    scenario_text = " ".join(
        [
            scenario.claim,
            scenario.source_name,
            scenario.source_type,
            scenario.region,
            scenario.rationale,
            " ".join(item.title for item in scenario.evidence),
        ]
    ).lower()
    words = [word for word in text.lower().split() if len(word) > 3]
    return sum(1 for word in words if word in scenario_text)


def _is_crisis_scope(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in CRISIS_SCOPE_TERMS)


def predict_claim(claim: str, source_type: str, include_media: bool = False) -> PredictionResult:
    text = f"{claim} {source_type}".strip()
    scenarios = get_scenarios()

    if not _is_crisis_scope(text):
        return PredictionResult(
            verdict=Verdict.INSUFFICIENT,
            confidence=0.22,
            risk_level="Out of scope",
            matched_scenario_id="none",
            difficulty_hint=Difficulty.EASY,
            explanation=(
                "This app is designed for war and crisis misinformation analysis. "
                "The entered claim does not contain enough crisis-related context, so the app should not guess a truth label."
            ),
            signals=[
                "No war or crisis context detected",
                "Claim needs live fact-checking or a general knowledge verifier",
            ],
            suggested_checks=[
                "Use a live fact source or trusted search for the latest status",
                "Add crisis context only if the claim is tied to a conflict or emergency event",
                "Avoid treating this output as a factual verdict for general-world claims",
            ],
            scope_status="out_of_scope",
        )

    best_scenario = max(scenarios, key=lambda scenario: _scenario_match_score(text, scenario))

    verdict_scores = {verdict: _token_score(text, keywords) for verdict, keywords in KEYWORD_RULES.items()}
    if include_media:
        verdict_scores[Verdict.MISLEADING] += 1
        verdict_scores[Verdict.FAKE] += 1

    top_verdict = max(verdict_scores, key=verdict_scores.get)
    if verdict_scores[top_verdict] == 0:
        top_verdict = best_scenario.verdict

    matched_signals = []
    for evidence in best_scenario.evidence:
        if any(word in evidence.title.lower() or word in evidence.summary.lower() for word in text.lower().split() if len(word) > 4):
            matched_signals.append(evidence.title)
    if not matched_signals:
        matched_signals = [item.title for item in best_scenario.evidence[:3]]

    confidence = 0.58 + min(0.3, verdict_scores[top_verdict] * 0.08)
    if top_verdict == best_scenario.verdict:
        confidence += 0.07
    confidence = min(0.96, round(confidence, 2))

    if confidence >= 0.85:
        risk_level = "High confidence"
    elif confidence >= 0.7:
        risk_level = "Moderate confidence"
    else:
        risk_level = "Needs verification"

    explanation = (
        f"This claim looks most similar to the `{best_scenario.id}` investigation pattern. "
        f"The app predicts `{top_verdict.value}` because the wording, source context, and signal pattern "
        f"overlap with known crisis misinformation or verification cases."
    )

    suggested_checks = [
        "Cross-check with trusted local or international reporters",
        "Verify the timeline and whether old media is being reused",
        "Inspect whether the source is official, anonymous, or spoofed",
    ]
    if include_media:
        suggested_checks.append("Run media forensics or reverse-image checks on the attached asset")

    return PredictionResult(
        verdict=top_verdict,
        confidence=confidence,
        risk_level=risk_level,
        matched_scenario_id=best_scenario.id,
        difficulty_hint=best_scenario.difficulty,
        explanation=explanation,
        signals=matched_signals,
        suggested_checks=suggested_checks,
        scope_status="crisis_scope",
    )
