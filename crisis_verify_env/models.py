from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Verdict(str, Enum):
    REAL = "real"
    MISLEADING = "misleading"
    FAKE = "fake"
    INSUFFICIENT = "insufficient_evidence"


class ActionType(str, Enum):
    INSPECT_SOURCE = "inspect_source"
    CHECK_TIMELINE = "check_timeline"
    CROSSCHECK_TRUSTED_REPORTS = "crosscheck_trusted_reports"
    ANALYZE_MEDIA = "analyze_media"
    SCAN_LANGUAGE = "scan_language"
    REQUEST_CONTEXT = "request_context"
    SUBMIT_VERDICT = "submit_verdict"


@dataclass
class EvidenceItem:
    id: str
    title: str
    summary: str
    supports_truth: bool
    useful_actions: list[ActionType]
    reliability: float
    distractor: bool = False


@dataclass
class Scenario:
    id: str
    difficulty: Difficulty
    claim: str
    source_name: str
    source_type: str
    region: str
    timestamp: str
    verdict: Verdict
    rationale: str
    evidence: list[EvidenceItem]
    max_steps: int = 6


@dataclass
class Action:
    action_type: ActionType
    argument: str | None = None
    verdict: Verdict | None = None
    confidence: float | None = None


@dataclass
class Observation:
    scenario_id: str
    difficulty: Difficulty
    claim: str
    source_name: str
    source_type: str
    region: str
    timestamp: str
    visible_evidence: list[str]
    discovered_evidence: list[str]
    action_history: list[str]
    steps_taken: int
    steps_remaining: int


@dataclass
class RewardBreakdown:
    investigation_reward: float = 0.0
    efficiency_penalty: float = 0.0
    verdict_reward: float = 0.0
    confidence_adjustment: float = 0.0
    total: float = 0.0


@dataclass
class StepInfo:
    explanation: str
    reward_breakdown: RewardBreakdown
    done_reason: Literal["continue", "submitted", "max_steps_reached"]


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: StepInfo


@dataclass
class EnvironmentState:
    scenario: Scenario
    steps_taken: int
    done: bool
    discovered_evidence_ids: list[str] = field(default_factory=list)
    action_history: list[str] = field(default_factory=list)
    submitted_verdict: Verdict | None = None
    submitted_confidence: float | None = None


def dataclass_to_dict(value: object) -> dict:
    return asdict(value)
