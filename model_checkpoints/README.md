# Model Checkpoints

This folder provides stable top-level links to the main saved checkpoints used to reproduce the paper tables and report-generation workflows.

## Paper Table Checkpoints

These are the primary best-model checkpoints for the classifier comparison table and the agentic workflow:

| Link | Model / use | Source run | Notes |
| --- | --- | --- | --- |
| `paper_simple_cnn_best.pth` | Simple CNN classifier | `experiments/repro_oc4_20260503_035316/simple_cnn_100` | Best checkpoint for the classifier metrics table |
| `paper_efficientnet_b3_best.pth` | EfficientNet-B3 classifier | `experiments/repro_oc4_20260503_035316/efficientnet_b3` | Best checkpoint for the classifier metrics table |
| `paper_resnet50_best.pth` | ResNet-50 classifier | `experiments/repro_oc4_20260503_035316/resnet50` | Best checkpoint for the classifier metrics table |
| `paper_vit_base_best.pth` | ViT-Base classifier | `experiments/repro_oc4_20260503_035316/vit_base` | Reproduced ViT checkpoint used in the classifier comparison table |
| `paper_swin_base_best.pth` | Swin classifier | `experiments/repro_oc4_20260503_035316/agent_swin` | Best classifier and agentic-review backbone |
| `paper_yolo_best.pt` | YOLO detector | `experiments/repro_oc4_20260503_035316/yolo/vinbig_yolov8m_e10` | Best YOLO checkpoint used in the comparison workflow |

The corresponding aggregate metrics are in:

- [summary.json](/project/community/sbandred/CT-transformer/experiments/repro_oc4_20260503_035316/summary.json)
- each model's `logs/training_history.json` under `experiments/repro_oc4_20260503_035316/`

## Additional Checkpoints

These are kept because they are referenced elsewhere in the repo or report discussion:

| Link | Model / use | Source run | Notes |
| --- | --- | --- | --- |
| `vit_base_100_epoch_best.pth` | ViT-Base 100-epoch run | `experiments/vit_base_100_2gpu` | Best 100-epoch ViT checkpoint discussed in the paper updates |
| `vit_cnn_sized_100_epoch_best.pth` | ViT-CNN-sized 100-epoch run | `experiments/vit_cnn_sized_100_2gpu` | Additional 100-epoch ViT-family run |
| `standalone_agent_swin_best.pth` | Swin review backbone | `experiments/agent_swin` | Standalone path used by the review config |
| `standalone_yolo_best.pt` | YOLO review backbone | `experiments/yolo_v8/full_e10local` | Standalone path used by the review config |

## Notes

- These files are symlinks, not copied checkpoints.
- If you move the underlying `experiments/` directories, these links must be regenerated.
- For exact paper-number verification, prefer the `paper_*` links above.
