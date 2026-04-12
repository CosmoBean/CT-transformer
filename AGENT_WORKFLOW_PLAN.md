# Agentic Workflow Plan

## Objective

Build a clinically relevant triage workflow on top of the trained Swin chest X-ray model.

## Workflow

1. Run Swin inference on a chest X-ray.
2. Convert logits to structured probabilities.
3. Route the case through a deterministic triage policy.
4. Return one of:
   - `accept_normal`
   - `accept_abnormal`
   - `flag_for_review`
   - `unable_to_assess`
5. Evaluate the workflow on the validation split.
6. Generate a confusion matrix and document representative cases from each bucket.

## Deliverables

- `src/agent/` package
- `scripts/demo_case.py`
- `scripts/evaluate_agent.py`
- `scripts/test_agent_workflow.py`
- `reports/agent/agent_eval.csv`
- `reports/agent/confusion_matrix.png`
- `reports/agent/confusion_matrix_summary.md`
- `reports/agent/case_review.md`

## Completion Criteria

- Single-case inference works with a trained Swin checkpoint.
- Agent decisions are deterministic and pass synthetic routing tests.
- Validation-set evaluation runs end-to-end.
- Accepted-case confusion matrix is generated.
- Representative true positive, true negative, false positive, false negative, and flagged cases are documented.
- Review bucket is enriched for harder cases relative to the accepted bucket.
