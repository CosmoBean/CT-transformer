# Reproduce Results Without Retraining

This repo now has one source-code-first entrypoint for the final workflow:

```bash
python main.py ...
```

The goal is simple:
- download prepared data from Hugging Face
- download checkpoints / cached artifacts from Hugging Face
- rerun comparisons without retraining
- generate a single agentic report with only an API key

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you want the API key to be picked up automatically, copy the template and fill it in:

```bash
cp .env.example .env
```

## 2. Download prepared assets

Assumptions:
- the processed dataset is published at [sbandred/vinbig-cxr-processed](https://huggingface.co/datasets/sbandred/vinbig-cxr-processed/tree/main) and contains the prepared `data/` contents
- the Hugging Face artifacts repo contains `experiments/` and checkpoint files laid out for this repo

```bash
python main.py download \
  --dataset-repo sbandred/vinbig-cxr-processed \
  --artifacts-repo <org_or_user/ct-transformer-artifacts>
```

## 3. Rerun comparisons without retraining

This reuses the downloaded checkpoints. If cached review JSON files are also present in the artifacts repo, the review comparison can rerun without hitting the model API.

```bash
python main.py compare --max-cases 300
```

If cached review outputs are missing and you want to recompute them:

```bash
python main.py compare \
  --max-cases 300 \
  --api-key <your_gateway_api_key>
```

Outputs land under:

```text
experiments/repro_outputs/
```

## 4. Generate one report from an API key

```bash
python main.py report \
  --image data/test/<image_id>.png \
  --api-key <your_gateway_api_key>
```

Outputs land under:

```text
experiments/agentic_reports/<image_id>/
```

## Under the hood

The CLI now calls the library workflows in `src/` directly, so the core usage path is one command surface instead of many wrapper scripts.
