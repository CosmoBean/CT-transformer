# CT-Transformer

Minimal code for three VinBigData workflows:

1. Classification
2. YOLO localization
3. Agentic report generation

## Setup

```bash
bash scripts/install.sh
source .venv/bin/activate
cp .env.example .env
```

Fill `.env` only if you need private Hugging Face access or agentic review generation.

## Reproduce without retraining

Download the prepared dataset and saved artifacts:

```bash
python scripts/reproduce.py download \
  --dataset-repo <hf_dataset_repo> \
  --artifacts-repo <hf_artifacts_repo>
```

Rerun the saved comparisons:

```bash
python scripts/reproduce.py compare --max-cases 300
```

Generate one report:

```bash
python scripts/reproduce.py report --image data/test/<image>.png
```

More detail is in [REPRODUCE.md](/project/community/sbandred/CT-transformer/REPRODUCE.md).

## Train from scratch

Classification:

```bash
python scripts/train.py --model swin_base_patch4_window7_224 --epochs 10
```

YOLO:

```bash
python scripts/train_yolo.py --weights yolov8m.pt
```

## Main commands

```bash
python scripts/train.py --help
python scripts/train_yolo.py --help
python scripts/evaluate_yolo.py --help
python scripts/evaluate_claude_review.py --help
python scripts/run_agentic_report.py --help
python scripts/reproduce.py --help
```

## Project layout

```text
configs/   runtime configuration
scripts/   thin CLI entrypoints
src/       library code
data/      prepared dataset
experiments/ checkpoints, reports, cached outputs
```
