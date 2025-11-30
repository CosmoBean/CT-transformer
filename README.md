# CT-Transformer: Chest X-ray Anomaly Detection

A comprehensive project for training state-of-the-art and creative approaches for chest X-ray anomaly detection.

## Features

### State-of-the-Art Models
- **Vision Transformer (ViT)** - Transformer-based classification
- **EfficientNet** - Efficient CNN architecture  
- **ResNet** - Deep residual networks
- **Swin Transformer** - Hierarchical vision transformer

### Creative Anomaly Detection Approaches
- **Autoencoder** - Reconstruction-based anomaly detection
- **Variational Autoencoder (VAE)** - Probabilistic reconstruction
- **AnoGAN** - GAN-based anomaly detection
- **Contrastive Learning** - Self-supervised representation learning

## Dataset

The project uses the VinBigData Chest X-ray dataset with:
- 18,000 postero-anterior chest X-ray scans
- 14 abnormality classes + "No finding"
- Images processed to 1024x1024 PNG format

See `DatasetInformation.md` for details.

## Installation

```bash
# Install dependencies
make install

# Or manually:
bash scripts/install.sh
```

## Usage

### Training a Model

```bash
# Using default config
python scripts/train.py

# Specify model
python scripts/train.py --model efficientnet_b3 --epochs 50

# Use custom config
python scripts/train.py --config configs/my_config.yaml
```

### Training in Jupyter Notebook

Open `notebooks/train_chest_xray_anomaly_detection.ipynb` for interactive training and experimentation.

## Project Structure

```
CT-transformer/
├── src/
│   ├── data/           # Dataset classes
│   ├── models/         # Model architectures
│   ├── training/       # Training utilities
│   └── utils/          # Utility functions
├── configs/            # Configuration files
├── notebooks/          # Jupyter notebooks
├── experiments/        # Training outputs
│   ├── checkpoints/    # Model checkpoints
│   └── logs/           # Training logs
├── scripts/            # Training scripts
└── data/               # Dataset (PNG images)
```

## Configuration

Edit `configs/default_config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data loading settings
- Device and paths

## Models

### Classification Models
- Multi-label classification for 15 classes (14 abnormalities + "No finding")
- Supports pretrained weights from ImageNet
- Flexible architecture selection

### Anomaly Detection Models
- Train on normal samples only
- Detect anomalies via reconstruction error
- Unsupervised learning approach

## Evaluation Metrics

- AUC-ROC (macro and micro)
- AUC-PR (Average Precision)
- F1 Score (macro and micro)
- Per-class metrics

## License

MIT License - see LICENSE file for details

