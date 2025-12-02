# CT-Transformer: Chest X-Ray Anomaly Detection

A comprehensive deep learning project for multi-label chest X-ray anomaly detection using state-of-the-art and novel hybrid transformer architectures.

## Overview

This project implements and evaluates multiple deep learning models for detecting abnormalities in chest X-ray images. The dataset contains 15,000 chest X-ray images with multi-label annotations for 14 different abnormalities plus "No finding" (normal).

## Key Achievements

### Model Performance Summary

Based on comprehensive training runs, here are the key results:

| Model | Accuracy | Hamming Accuracy | AUC-ROC (macro) | F1 (macro) | Training Time |
|-------|----------|------------------|-----------------|------------|---------------|
| **Swin Transformer** | **0.7517** | **0.9592** | **0.9622** | **0.5604** | 21.8 min |
| EfficientNet-B3 | 0.7457 | 0.9536 | 0.9488 | 0.4437 | 19.1 min |
| ResNet-50 | 0.7463 | 0.9541 | 0.9477 | 0.3850 | 19.0 min |
| Vision Transformer | 0.7007 | 0.9269 | 0.8143 | 0.1314 | 19.9 min |
| FLARE | 0.6973 | 0.9189 | 0.7570 | 0.0987 | 18.5 min |

**Best Performing Model**: Swin Transformer achieves the highest accuracy (75.17%) and AUC-ROC (96.22%), making it the top choice for this task.

### Novel Hybrid Models

Three innovative hybrid architectures have been developed that combine different techniques:

1. **FLARE Hybrid Classifier** (~153M parameters)
   - Combines CNN backbone (EfficientNet) with FLARE transformer
   - Uses cross-attention fusion to integrate local CNN features with global FLARE attention
   - Architecture: CNN extracts local patterns → FLARE processes global context → Cross-attention fusion

2. **Multi-Scale FLARE** (~177M parameters)
   - Processes images at multiple resolutions (256x256 and 512x512)
   - Uses cross-scale attention for feature fusion
   - Learnable scale weights for adaptive multi-scale combination
   - Better captures both fine-grained details and broader context

3. **FLARE with Attention Pooling** (~210M parameters)
   - Replaces class token with multi-head attention pooling
   - Adaptive feature aggregation that learns to focus on relevant regions
   - More expressive than standard mean pooling or class token

### Technical Improvements

1. **Data Pipeline**
   - Proper train/validation split (80/20) from training data
   - Reproducible splits with fixed random seed
   - Support for 512x512 image resolution (original: 1024x1024)
   - Multi-label classification with 15 classes

2. **Evaluation Metrics**
   - Exact match accuracy (strict: all labels must be correct)
   - Hamming accuracy (per-label accuracy, more lenient)
   - AUC-ROC (macro and micro) for ranking quality
   - AUC-PR (Average Precision)
   - F1-score (macro and micro)
   - Proper metric calculation for autoencoder/VAE models via reconstruction error

3. **Multi-GPU Support**
   - DataParallel support for all models
   - Automatic batch size scaling across GPUs
   - Fixed CUDA memory alignment issues for nested modules

4. **Anomaly Detection Models**
   - Autoencoder and VAE with proper metric calculation
   - Reconstruction error converted to anomaly scores
   - Binary anomaly labels (normal vs. anomalous) for evaluation

## Features

### State-of-the-Art Models
- **Vision Transformer (ViT)** - Pure transformer architecture
- **EfficientNet-B3** - Efficient CNN with compound scaling
- **ResNet-50** - Deep residual networks
- **Swin Transformer** - Hierarchical vision transformer with shifted windows
- **FLARE** - Efficient linear transformer with latent attention

### Novel Hybrid Architectures
- **FLARE Hybrid** - CNN + FLARE with cross-attention fusion
- **Multi-Scale FLARE** - Multi-resolution processing with cross-scale attention
- **FLARE Attention Pooling** - Adaptive attention-based feature aggregation

### Anomaly Detection Models
- **Autoencoder** - Reconstruction-based anomaly detection
- **Variational Autoencoder (VAE)** - Probabilistic reconstruction

## Dataset

The project uses the VinBigData Chest X-ray dataset:
- **15,000** chest X-ray images (12,000 train / 3,000 validation)
- **15 classes**: 14 abnormality types + "No finding" (normal)
- **Image size**: 1024x1024 (processed to 512x512 for training)
- **Format**: PNG images with CSV annotations

See `DatasetInformation.md` for detailed class information.

## Installation

### Prerequisites
- Python 3.11+
- CUDA-capable GPU (recommended)
- 20GB+ disk space for dataset and models

### Quick Install

```bash
# Install all dependencies
make install

# Or manually:
bash scripts/install.sh
```

This will:
- Create a virtual environment
- Install PyTorch 2.7 with CUDA support
- Install all required packages (timm, transformers, scikit-learn, etc.)

## Usage

### Training Individual Models

```bash
# Train a specific model
make train-efficientnet    # EfficientNet-B3
make train-resnet          # ResNet-50
make train-vit             # Vision Transformer
make train-swin            # Swin Transformer
make train-flare           # FLARE transformer

# Train hybrid models
make train-flare-hybrid        # FLARE + CNN hybrid
make train-flare-multiscale    # Multi-scale FLARE
make train-flare-attn-pool     # FLARE with attention pooling

# Train anomaly detection models
make train-autoencoder     # Autoencoder
make train-vae            # Variational Autoencoder
```

### Training All Models

```bash
# Train all models sequentially
make train-all

# Train all models in background (for overnight runs)
make train-all-bg

# View results
make view-results
```

### Quick Testing (1 epoch)

```bash
make test-efficientnet
make test-resnet
make test-vit
make test-swin
make test-flare
make test-flare-hybrid
make test-flare-multiscale
make test-flare-attn-pool
```

### Custom Training

```bash
# Using default config
python scripts/train.py

# Specify model and epochs
python scripts/train.py --model efficientnet_b3 --epochs 50

# Use custom config
python scripts/train.py --config configs/my_config.yaml
```

## Project Structure

```
CT-transformer/
├── src/
│   ├── data/              # Dataset classes and data loaders
│   │   ├── dataset.py      # ChestXRayDataset, AnomalyDetectionDataset
│   │   └── transforms.py  # Data augmentation
│   ├── models/            # Model architectures
│   │   ├── sota_models.py      # SOTA models (ViT, EfficientNet, ResNet, Swin)
│   │   ├── flare_model.py      # FLARE transformer implementation
│   │   ├── hybrid_models.py    # Novel hybrid architectures
│   │   └── anomaly_models.py   # Autoencoder, VAE
│   ├── training/          # Training utilities
│   │   ├── trainer.py     # Main training loop
│   │   └── metrics.py     # Evaluation metrics
│   └── utils/             # Utility functions
│       └── config.py      # Configuration loading
├── configs/               # Configuration files
│   └── default_config.yaml
├── scripts/               # Training and utility scripts
│   ├── train.py           # Main training script
│   ├── train_all_models.py # Batch training orchestrator
│   ├── view_results.py    # Results viewer
│   ├── test_setup.py      # Setup validation
│   └── validate_setup.py # Comprehensive validation
├── experiments/           # Training outputs
│   ├── checkpoints/      # Model checkpoints
│   ├── logs/             # Training logs
│   └── model_results.json # Training results
├── results/              # Results summaries
│   ├── model_results_accuracy.json
│   └── model_results_auc.json
├── data/                 # Dataset directory
│   ├── train/            # Training images
│   ├── test/             # Test images
│   └── train.csv         # Annotations
├── makefile              # Build commands
└── README.md             # This file
```

## Configuration

Edit `configs/default_config.yaml` to customize:

### Data Configuration
- `image_size`: Image resolution (default: 512)
- `batch_size`: Batch size (default: 32)
- `train_split`: Training fraction (default: 0.8)
- `val_split`: Validation fraction (default: 0.2)

### Model Configuration
- `name`: Model architecture
- `num_classes`: Number of classes (15)
- `pretrained`: Use pretrained weights
- `embed_dim`: Embedding dimension (for transformers)
- `depth`: Number of transformer layers
- `num_heads`: Number of attention heads
- `num_latents`: Number of latent queries (for FLARE)

### Training Configuration
- `num_epochs`: Number of training epochs (default: 50)
- `learning_rate`: Learning rate (default: 1e-4)
- `optimizer`: Optimizer (adam, adamw, sgd)
- `scheduler`: Learning rate scheduler (cosine, step, plateau)
- `metric_name`: Primary metric for model selection (accuracy, auc_roc_macro)

## Evaluation Metrics

The project supports comprehensive evaluation metrics:

### Classification Metrics
- **Accuracy**: Exact match accuracy (all labels must be correct)
- **Hamming Accuracy**: Per-label accuracy (more lenient)
- **AUC-ROC**: Area under ROC curve (macro and micro)
- **AUC-PR**: Average precision (macro and micro)
- **F1-Score**: F1 score (macro and micro)

### Anomaly Detection Metrics
- **Reconstruction Error**: MSE between input and reconstruction
- **Anomaly Score**: Normalized reconstruction error
- **Binary Classification Metrics**: Accuracy, AUC-ROC, F1 for normal vs. anomalous

## Model Architectures

### Standard Models

1. **EfficientNet-B3**: Efficient CNN with compound scaling
   - Parameters: ~12M
   - Best for: Fast inference, good accuracy

2. **ResNet-50**: Deep residual network
   - Parameters: ~25M
   - Best for: Proven architecture, stable training

3. **Vision Transformer (ViT-Base)**: Pure transformer
   - Parameters: ~86M
   - Best for: Large-scale pretraining, global attention

4. **Swin Transformer**: Hierarchical vision transformer
   - Parameters: ~88M
   - Best for: Best accuracy, efficient attention

5. **FLARE**: Linear transformer with latent attention
   - Parameters: ~210M (with increased capacity)
   - Best for: Efficient attention, linear complexity

### Hybrid Models

1. **FLARE Hybrid**: CNN + FLARE with cross-attention
   - Combines local CNN features with global FLARE attention
   - Cross-attention fusion for feature integration

2. **Multi-Scale FLARE**: Multi-resolution processing
   - Processes at 256x256 and 512x512
   - Cross-scale attention for feature fusion

3. **FLARE Attention Pooling**: Adaptive pooling
   - Multi-head attention pooling instead of class token
   - Learns to focus on relevant image regions

### Anomaly Detection Models

1. **Autoencoder**: Reconstruction-based
   - Encoder-decoder architecture
   - Anomaly score = reconstruction error

2. **Variational Autoencoder (VAE)**: Probabilistic reconstruction
   - Encoder-decoder with latent distribution
   - Anomaly score = reconstruction error + KL divergence

## Training Tips

1. **Multi-GPU Training**: Automatically enabled if multiple GPUs are available
   - Batch size scales with number of GPUs
   - Use `use_multi_gpu: true` in config

2. **Image Size**: 512x512 is a good balance between quality and memory
   - Original images are 1024x1024
   - Larger sizes require more GPU memory

3. **Learning Rate**: Start with 1e-4 and adjust based on validation performance
   - Use cosine annealing for smooth decay
   - Reduce on plateau if validation stalls

4. **Early Stopping**: Configured via `early_stopping_patience` in config
   - Default: 10 epochs
   - Saves best model based on primary metric

## Results and Logging

### Training Results
- Results saved to `experiments/model_results.json`
- Includes metrics, training time, and status for each model
- View with: `make view-results` or `python scripts/view_results.py`

### Checkpoints
- Best models saved to `experiments/checkpoints/`
- Named by model and metric: `{model_name}_best_{metric}.pth`
- Latest checkpoint: `latest_model.pth`

### Logs
- Training logs: `experiments/logs/`
- Batch training log: `experiments/training_log.txt`
- Background training: `experiments/training_nohup.log`

## Testing and Validation

```bash
# Test project setup
make test

# Comprehensive model validation
make test-models

# Quick 1-epoch test of any model
make test-efficientnet
make test-flare-hybrid
# etc.
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `image_size` (e.g., 256 instead of 512)
- Use gradient accumulation

### DataParallel Issues
- All hybrid models have been fixed for DataParallel compatibility
- Ensure tensors are contiguous (handled automatically)

### Model Not Training
- Check learning rate (try 1e-5 or 1e-3)
- Verify data loading (run `make test`)
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

## Future Work

- [ ] Distributed DataParallel (DDP) for better multi-GPU scaling
- [ ] Additional hybrid architectures
- [ ] Ensemble methods
- [ ] Test-time augmentation
- [ ] Model interpretability (attention visualization)
- [ ] Deployment optimization (quantization, ONNX export)

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ct_transformer,
  title={CT-Transformer: Chest X-Ray Anomaly Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/CT-transformer}
}
```

## Acknowledgments

- VinBigData for the chest X-ray dataset
- PyTorch team for the deep learning framework
- `timm` library for pretrained vision models
- FLARE authors for the efficient transformer architecture
