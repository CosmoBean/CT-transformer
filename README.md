# CT-Transformer: Chest X-Ray Anomaly Detection

Chest X-ray anomaly detection on VinBigData using strong image backbones for multi-label classification and reconstruction-based anomaly detection.

## Supported Models

- `efficientnet_b3`
- `resnet50`
- `vit_base`
- `swin_base_patch4_window7_224`
- `autoencoder`
- `vae`

## Current Classification Results

Historical classifier summaries in `results/` currently show:

| Model | Accuracy | Hamming Accuracy | AUC-ROC (macro) | F1 (macro) |
| --- | --- | --- | --- | --- |
| Swin Transformer | 0.7517 | 0.9592 | 0.9622 | 0.5604 |
| EfficientNet-B3 | 0.7457 | 0.9536 | 0.9488 | 0.4437 |
| ResNet-50 | 0.7463 | 0.9541 | 0.9477 | 0.3850 |
| Vision Transformer | 0.7007 | 0.9269 | 0.8143 | 0.1314 |

Fresh corrected reruns on `oc4` so far have Swin still leading on macro AUC.

## Dataset

The project uses the VinBigData Chest X-ray dataset:

- `15,000` images
- `15` labels: `14` abnormalities plus `No finding`
- PNG images with CSV annotations
- Training images prepared under `data/train`
- Test images prepared under `data/test`

## Setup

```bash
make install
make data
```

That creates `.venv`, installs dependencies, and prepares the dataset from the local CSV / downloaded assets.

## Training

Train a single classifier:

```bash
python scripts/train.py --model swin_base_patch4_window7_224 --epochs 10
```

Useful `make` targets:

```bash
make train-efficientnet
make train-resnet
make train-vit
make train-swin
make train-autoencoder
make train-vae
make train-all
```

Quick smoke tests:

```bash
make test
make test-models
make test-efficientnet
make test-resnet
make test-vit
make test-swin
```

## Configuration

Main config: `configs/default_config.yaml`

Important fields:

- `data.image_size`
- `data.batch_size`
- `data.mode`
- `model.name`
- `model.pretrained`
- `training.num_epochs`
- `training.learning_rate`
- `training.metric_name`

For anomaly models, `scripts/train.py` automatically switches to anomaly mode and binary `auc_roc` selection.

## Outputs

- Checkpoints: `experiments/checkpoints/`
- Logs: `experiments/logs/`
- Batch summaries: `experiments/model_results.json`
- Historical summaries: `results/model_results_accuracy.json`, `results/model_results_auc.json`

## Notes

- Best-checkpoint selection now uses `auc_roc_macro` for multi-label classification by default.
- Anomaly labels now correctly exclude the `No finding` column when computing abnormal vs. normal targets.
- Autoencoder and VAE metrics from older historical result files are stale and should be refreshed from the corrected reruns.
