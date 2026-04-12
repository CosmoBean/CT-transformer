"""
Structured schemas for the agentic triage workflow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class AgentAction(str, Enum):
    ACCEPT_NORMAL = "accept_normal"
    ACCEPT_ABNORMAL = "accept_abnormal"
    FLAG_FOR_REVIEW = "flag_for_review"
    UNABLE_TO_ASSESS = "unable_to_assess"


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CaseSummary:
    image_id: str
    image_path: str
    class_probabilities: dict[str, float]
    top_findings: list[str]
    top_scores: dict[str, float]
    no_finding_probability: float
    max_abnormal_probability: float
    abnormal_probability_margin: float
    moderate_findings_count: int
    heuristic_binary_prediction: str
    valid_image: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentDecision:
    image_id: str
    action: AgentAction
    confidence: ConfidenceBand
    predicted_findings: list[str]
    recommendation: str
    rationale: str
    heuristic_binary_prediction: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["confidence"] = self.confidence.value
        return payload
