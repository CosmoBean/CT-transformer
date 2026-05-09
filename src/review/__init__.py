"""
Claude-backed review workflow for multimodel chest X-ray decision support.
"""

from .builder import build_case_packet, load_prompt_template, render_user_prompt
from .gateway import CmuGatewayClient, GatewayError
from .inference import SwinInferenceEngine, YoloInferenceEngine
from .metrics import build_case_buckets, evaluate_multilabel_predictions
from .orchestrator import ReviewOrchestrator
from .renderer import render_review_report
from .schema import ClaudeReviewResponse, validate_review_response
from .taxonomy import DERIVED_GLOBAL_BUCKETS, LABEL_CATEGORY_BY_NAME, derive_global_bucket_hints
from .workflows import build_review_orchestrator, evaluate_review_run, run_review_case

__all__ = [
    "build_case_packet",
    "load_prompt_template",
    "render_user_prompt",
    "CmuGatewayClient",
    "GatewayError",
    "SwinInferenceEngine",
    "YoloInferenceEngine",
    "build_case_buckets",
    "evaluate_multilabel_predictions",
    "ReviewOrchestrator",
    "build_review_orchestrator",
    "evaluate_review_run",
    "run_review_case",
    "render_review_report",
    "ClaudeReviewResponse",
    "validate_review_response",
    "DERIVED_GLOBAL_BUCKETS",
    "LABEL_CATEGORY_BY_NAME",
    "derive_global_bucket_hints",
]
