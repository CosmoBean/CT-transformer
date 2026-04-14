# CT-Transformer

Simple chest X-ray training code for the VinBigData dataset.

The repo has three main parts:
- dataset loading in `src/data`
- model definitions in `src/models`
- training and metrics in `src/training`

## Models

Classification:
- `efficientnet_b3`
- `resnet50`
- `vit_base`
- `swin_base_patch4_window7_224`

Anomaly detection:
- `autoencoder`
- `vae`

## Dataset

The project expects the VinBigData chest X-ray dataset in `data/`.

Prepared layout:

```text
data/
├── train/
├── test/
└── train.csv
```

## Setup

Create the virtual environment and install dependencies:

```bash
make install
```

Prepare the dataset:

```bash
make data
```

## Train

Train one model:

```bash
python scripts/train.py --model swin_base_patch4_window7_224 --epochs 10
```

Or use the short `make` targets:

```bash
make train-efficientnet
make train-resnet
make train-vit
make train-swin
make train-autoencoder
make train-vae
```

Train every supported model:

```bash
make train-all
```

## Validate

Basic setup checks:

```bash
make test
make test-models
```

## Files

- config: `configs/default_config.yaml`
- dataset code: `src/data/dataset.py`
- transforms: `src/data/transforms.py`
- model code: `src/models/`
- trainer: `src/training/trainer.py`
- training entrypoint: `scripts/train.py`

## Results

Older summary files are kept in `results/` for reference:

- `results/model_results_accuracy.json`
- `results/model_results_auc.json`
