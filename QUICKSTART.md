# Quick Start Guide

## 1. Setup Environment

```bash
# Install dependencies
make install

# Activate virtual environment
source .venv/bin/activate
```

## 2. Verify Data Structure

Your data should be organized as:
```
data/
├── train/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── test/
│   └── ...
└── train_meta.csv  # Optional: annotations
```

## 3. Train a Model

### Option A: Using the Training Script

```bash
# Train EfficientNet (recommended for first run)
python scripts/train.py --model efficientnet_b3 --epochs 20

# Train Vision Transformer
python scripts/train.py --model vit_base --epochs 30

# Train Autoencoder for anomaly detection
python scripts/train.py --model autoencoder --epochs 20
```

### Option B: Using Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/train_chest_xray_anomaly_detection.ipynb
```

## 4. Available Models

### Classification Models:
- `efficientnet_b3` - Fast and effective (recommended)
- `resnet50` - Classic CNN
- `vit_base` - Vision Transformer
- `swin_base` - Swin Transformer

### Anomaly Detection Models:
- `autoencoder` - Reconstruction-based
- `vae` - Variational Autoencoder

## 5. Monitor Training

Check training progress:
```bash
# View logs
tail -f experiments/logs/*/training_history.json

# Check checkpoints
ls experiments/checkpoints/*/
```

## 6. Customize Configuration

Edit `configs/default_config.yaml` to adjust:
- Image size
- Batch size
- Learning rate
- Model architecture
- Training epochs

## Example Training Commands

```bash
# Quick test run (5 epochs)
python scripts/train.py --model efficientnet_b3 --epochs 5

# Full training with custom config
python scripts/train.py --config configs/default_config.yaml

# Train multiple models
for model in efficientnet_b3 resnet50 vit_base; do
    python scripts/train.py --model $model --epochs 20
done
```

## Troubleshooting

### Out of Memory
- Reduce batch size in config: `batch_size: 16`
- Use smaller image size: `image_size: 128`

### Slow Training
- Reduce number of workers: `num_workers: 2`
- Use smaller model: `efficientnet_b0` instead of `efficientnet_b3`

### No Annotations
The code works without annotations (CSV file). It will create dummy labels for testing.

