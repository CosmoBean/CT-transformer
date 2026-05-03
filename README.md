# CT-Transformer

Clean chest X-ray code for the **VinBigData** dataset with three retained paths:

1. **Classification**
   - from-scratch baseline: `simple_cnn`
   - transfer-learning baselines: `efficientnet_b3`, `resnet50`, `vit_base`, `swin_base_patch4_window7_224`
2. **Localization**
   - `YOLOv8m` trained on the raw VinBigData box annotations
3. **Agentic reporting**
   - `Swin + YOLO + agentic AI reviewer` to generate a structured decision-support report

This repo is intentionally scoped to the final workflow. Older anomaly-detection and dead experimental branches were removed.

## Dataset

The project uses the current **15-label VinBigData subset** already prepared in:

```text
data/
├── train/
├── test/
└── train.csv
```

For YOLO localization, the raw box annotations are also required:

```text
data/_downloads/train_raw.csv
data/_downloads/vinbig_png/train_meta.csv
```

`scripts/setup_data.py` is the main dataset setup entrypoint. If you later upload this prepared dataset to Hugging Face, you can keep the same folder structure and just swap the download/source step.

## Final repo structure

```text
configs/
  default_config.yaml          # classification defaults
  yolo_v8_detection.yaml       # YOLO defaults
  claude_review.yaml           # agentic review defaults

src/
  data/                        # VinBigData classification + detection dataset code
  models/                      # simple CNN + transfer-learning classifiers
  training/                    # trainer + metrics
  review/                      # agentic reporting workflow

scripts/
  setup_data.py                # dataset preparation
  train.py                     # classification training
  prepare_yolo_dataset.py      # YOLO dataset materialization
  train_yolo.py                # YOLO training
  run_agentic_report.py        # end-to-end report generation
```

## Setup

Install dependencies:

```bash
make install
```

Prepare the dataset:

```bash
make data
```

## Classification training

The default classification config is fixed in [configs/default_config.yaml](/project/community/sbandred/CT-transformer/configs/default_config.yaml).

Train the from-scratch baseline:

```bash
python scripts/train.py --model simple_cnn --epochs 100 \
  --save-dir experiments/simple_cnn_100/checkpoints \
  --log-dir experiments/simple_cnn_100/logs
```

Train a transfer-learning model:

```bash
python scripts/train.py --model swin_base_patch4_window7_224 --epochs 10 \
  --save-dir experiments/agent_swin/checkpoints \
  --log-dir experiments/agent_swin/logs
```

Shortcuts:

```bash
make train-simple-cnn
make train-efficientnet
make train-resnet
make train-vit
make train-swin
```

## YOLO localization

Prepare the YOLO dataset:

```bash
make prepare-yolo
```

Train YOLOv8m:

```bash
make train-yolo
```

Optional single-image inference:

```bash
make infer-yolo
```

Optional YOLO evaluation:

```bash
make eval-yolo
```

## Agentic report generation

The final report path is:

```text
CXR image -> Swin classification -> YOLO localization -> agentic AI review -> PDF/Markdown/JSON report bundle
```

Run the full report workflow on one image:

```bash
export CMU_LLM_GATEWAY_API_KEY=...
python scripts/run_agentic_report.py --image path/to/cxr.png
```

Or use:

```bash
make agentic-report
```

## Retained best artifacts

Important current experiment directories:

- `experiments/agent_swin/`
- `experiments/simple_cnn_100/`
- `experiments/yolo_v8/`
- `experiments/claude_review/`

Presentation-ready report PDFs are kept separately in:

- `reports/presentation_comparison_examples/`

## Sanity checks

```bash
make test
make test-yolo
make test-review
```
