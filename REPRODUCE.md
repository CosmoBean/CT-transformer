# Reproduce Results Without Retraining

This repo now has one source-code-first entrypoint for the final workflow:

```bash
python scripts/reproduce.py ...
```

The goal is simple:
- download prepared data from Hugging Face
- download checkpoints / cached artifacts from Hugging Face
- rerun comparisons without retraining
- generate a single agentic report with only an API key

## 1. Install

```bash
bash scripts/install.sh
source .venv/bin/activate
```

## 2. Download prepared assets

Assumptions:
- the Hugging Face dataset repo contains the prepared `data/` contents
- the Hugging Face artifacts repo contains `experiments/` and checkpoint files laid out for this repo

```bash
python scripts/reproduce.py download \
  --dataset-repo <org_or_user/vinbigdata-prepared> \
  --artifacts-repo <org_or_user/ct-transformer-artifacts>
```

If either repo is private:

```bash
python scripts/reproduce.py download \
  --dataset-repo <org_or_user/vinbigdata-prepared> \
  --artifacts-repo <org_or_user/ct-transformer-artifacts> \
  --hf-token <your_hf_token>
```

## 3. Rerun comparisons without retraining

This reuses the downloaded checkpoints. If cached review JSON files are also present in the artifacts repo, the review comparison can rerun without hitting the model API.

```bash
python scripts/reproduce.py compare --max-cases 300
```

If cached review outputs are missing and you want to recompute them:

```bash
python scripts/reproduce.py compare \
  --max-cases 300 \
  --api-key <your_gateway_api_key>
```

Outputs land under:

```text
experiments/repro_outputs/
```

## 4. Generate one report from an API key

```bash
python scripts/reproduce.py report \
  --image data/test/<image_id>.png \
  --api-key <your_gateway_api_key>
```

Outputs land under:

```text
experiments/agentic_reports/<image_id>/
```

## 5. Generate presentation comparison reports

If you already have the comparison CSV and cache bundle:

```bash
python scripts/reproduce.py presentation
```

## Under the hood

The wrapper intentionally keeps the existing scripts intact:
- `scripts/evaluate_yolo.py`
- `scripts/evaluate_claude_review.py`
- `scripts/run_agentic_report.py`
- `scripts/generate_presentation_comparison_reports.py`

The difference is that you no longer need to remember the individual command chain.
