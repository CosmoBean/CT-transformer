# CT-Transformer

Minimal code for three VinBigData workflows:

1. Classification
2. YOLO localization
3. Agentic report generation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` only if you need agentic review generation.

## Reproduce without retraining

The default path in this repo is:
- download prepared data and saved artifacts
- rerun the comparisons without retraining
- generate reports from cached outputs or from the API key

This is the fastest way to reproduce the final results.

### 1. Download prepared assets

Assumptions:
- the processed dataset is published at [sbandred/vinbig-cxr-processed](https://huggingface.co/datasets/sbandred/vinbig-cxr-processed/tree/main) and contains the prepared `data/` contents

Download the processed dataset:

```bash
python main.py download \
  --dataset-repo sbandred/vinbig-cxr-processed
```

### 2. Rerun comparisons without retraining

This reuses the downloaded checkpoints. If cached review JSON files are present in the artifacts repo, the review comparison reruns without calling the model API.

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

### 3. Generate one report

Generate one agentic report directly from an image:

```bash
python main.py report --image data/test/<image>.png
```

If you want to force regeneration through the gateway:

```bash
python main.py report \
  --image data/test/<image>.png \
  --api-key <your_gateway_api_key>
```

Outputs land under:

```text
experiments/agentic_reports/<image_id>/
```

### 4. Generate the report templates

These scripts regenerate the two PDF templates used in this repo from cached review outputs:

```bash
python scripts/generate_reports.py --list-templates
python scripts/generate_reports.py --template example
python scripts/generate_reports.py --template comparison
```

Template outputs:
- `example`: doctor-facing report without ground-truth comparison
- `comparison`: presentation report comparing pipeline output against ground truth

Generated PDFs land under:

```text
reports/example_reports/
reports/comparision_reports/
```

Best checkpoints for the paper tables are indexed under:

```text
model_checkpoints/
```

### 5. API-key-only regeneration

The PDF templates are reproducible from cached artifacts today.

A fresh user with only an API key can reproduce the full path by:
1. downloading the public data and artifacts
2. running `python main.py compare --api-key ...` if review cache is missing
3. running `python scripts/generate_reports.py --template example` or `--template comparison`

## Train from scratch

Classification:

```bash
python main.py train-classifier --model swin_base_patch4_window7_224 --epochs 10
```

YOLO:

```bash
python main.py train-yolo --weights yolov8m.pt
```

## Main commands

```bash
python main.py --help
```

## Project layout

```text
configs/   runtime configuration
main.py    primary CLI
scripts/   install and auxiliary utilities
src/       library code
tests/     test suite
data/      prepared dataset
experiments/ checkpoints, reports, cached outputs
```
