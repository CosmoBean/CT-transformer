# Quick Start

## 1. Install

```bash
make install
source .venv/bin/activate
```

## 2. Prepare Data

```bash
make data
```

Expected layout:

```text
data/
├── train/
├── test/
└── train.csv
```

## 3. Train

```bash
python scripts/train.py --model efficientnet_b3 --epochs 10
```

Other model names:
- `resnet50`
- `vit_base`
- `swin_base_patch4_window7_224`
- `autoencoder`
- `vae`

## 4. Validate

```bash
make test
make test-models
```

## 5. Results

Training writes outputs under `experiments/`.
