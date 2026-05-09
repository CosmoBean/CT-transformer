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
python main.py download \
  --dataset-repo <hf_dataset_repo> \
  --artifacts-repo <hf_artifacts_repo>
```

Rerun the saved comparisons:

```bash
python main.py compare --max-cases 300
```

Generate one report:

```bash
python main.py report --image data/test/<image>.png
```

More detail is in [REPRODUCE.md](/project/community/sbandred/CT-transformer/REPRODUCE.md).

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
