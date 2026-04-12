"""
Deterministic triage policy on top of model probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import AgentAction, AgentDecision, CaseSummary, ConfidenceBand


@dataclass(frozen=True)
class AgentPolicyConfig:
    normal_accept_threshold: float = 0.88
    normal_max_abnormal_threshold: float = 0.12
    abnormal_accept_threshold: float = 0.75
    abnormal_max_no_finding_threshold: float = 0.25
    abnormal_margin_threshold: float = 0.18
    review_margin_threshold: float = 0.05
    review_moderate_threshold: float = 0.35
    review_multi_findings_threshold: int = 1


class AgentTriagePolicy:
    """
    Small, inspectable routing policy for clinical triage support.
    """

    def __init__(self, config: AgentPolicyConfig | None = None):
        self.config = config or AgentPolicyConfig()

    def decide(self, summary: CaseSummary) -> AgentDecision:
        if not summary.valid_image:
            return AgentDecision(
                image_id=summary.image_id,
                action=AgentAction.UNABLE_TO_ASSESS,
                confidence=ConfidenceBand.LOW,
                predicted_findings=[],
                recommendation="Image could not be processed. Route for manual review.",
                rationale=summary.error or "Image preprocessing failed.",
                heuristic_binary_prediction="unknown",
            )

        no_finding = summary.no_finding_probability
        max_abnormal = summary.max_abnormal_probability
        margin = summary.abnormal_probability_margin
        moderate_count = self._moderate_findings_count(summary)
        predicted_findings = summary.top_findings[:2]

        if (
            no_finding >= self.config.normal_accept_threshold
            and max_abnormal <= self.config.normal_max_abnormal_threshold
        ):
            confidence = self._confidence_from_scores(no_finding, max_abnormal, margin)
            return AgentDecision(
                image_id=summary.image_id,
                action=AgentAction.ACCEPT_NORMAL,
                confidence=confidence,
                predicted_findings=["No finding"],
                recommendation="Likely normal chest X-ray. Low-priority review is reasonable.",
                rationale=(
                    f"No-finding probability is high ({no_finding:.2f}) and all abnormal scores "
                    f"remain low (max abnormal {max_abnormal:.2f})."
                ),
                heuristic_binary_prediction="normal",
            )

        if (
            max_abnormal >= self.config.abnormal_accept_threshold
            and no_finding <= self.config.abnormal_max_no_finding_threshold
            and margin >= self.config.abnormal_margin_threshold
            and moderate_count <= self.config.review_multi_findings_threshold
        ):
            confidence = self._confidence_from_scores(max_abnormal, no_finding, margin)
            return AgentDecision(
                image_id=summary.image_id,
                action=AgentAction.ACCEPT_ABNORMAL,
                confidence=confidence,
                predicted_findings=predicted_findings,
                recommendation="Likely abnormal chest X-ray. Prioritize radiologist review.",
                rationale=(
                    f"Top abnormal finding is strong ({max_abnormal:.2f}), no-finding is lower "
                    f"({no_finding:.2f}), and the abnormal margin is clear ({margin:.2f})."
                ),
                heuristic_binary_prediction="abnormal",
            )

        return AgentDecision(
            image_id=summary.image_id,
            action=AgentAction.FLAG_FOR_REVIEW,
            confidence=ConfidenceBand.LOW if margin < self.config.review_margin_threshold else ConfidenceBand.MEDIUM,
            predicted_findings=predicted_findings,
            recommendation="Uncertain case. Send for closer human review.",
            rationale=self._review_rationale(summary),
            heuristic_binary_prediction=summary.heuristic_binary_prediction,
        )

    def _confidence_from_scores(self, primary_score: float, competing_score: float, margin: float) -> ConfidenceBand:
        if primary_score >= 0.85 and margin >= 0.20 and abs(primary_score - competing_score) >= 0.35:
            return ConfidenceBand.HIGH
        if primary_score >= 0.65 and margin >= 0.10:
            return ConfidenceBand.MEDIUM
        return ConfidenceBand.LOW

    def _review_rationale(self, summary: CaseSummary) -> str:
        parts = []
        moderate_count = self._moderate_findings_count(summary)
        if summary.abnormal_probability_margin < self.config.review_margin_threshold:
            parts.append(
                f"Top abnormal findings are close together (margin {summary.abnormal_probability_margin:.2f})."
            )
        if summary.no_finding_probability > 0.40 and summary.max_abnormal_probability > 0.40:
            parts.append(
                f"Normal ({summary.no_finding_probability:.2f}) and abnormal ({summary.max_abnormal_probability:.2f}) signals conflict."
            )
        if moderate_count > self.config.review_multi_findings_threshold:
            parts.append(
                f"Several moderate findings are present ({moderate_count} classes above review threshold)."
            )
        if not parts:
            parts.append("Prediction confidence is not strong enough for automatic acceptance.")
        return " ".join(parts)

    def _moderate_findings_count(self, summary: CaseSummary) -> int:
        if not summary.class_probabilities:
            return summary.moderate_findings_count
        return sum(
            prob >= self.config.review_moderate_threshold
            for class_name, prob in summary.class_probabilities.items()
            if class_name != "No finding"
        )
