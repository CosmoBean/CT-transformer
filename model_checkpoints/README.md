# Model Checkpoints

This folder provides stable top-level links to the main saved checkpoints used to reproduce the paper tables and report-generation workflows.

## Paper Table Checkpoints

These are the primary best-model checkpoints for the classifier comparison table and the agentic workflow:

| Link | Model / use | Notes |
| --- | --- | --- |
| `paper_simple_cnn_best.pth` | Simple CNN classifier | Best checkpoint for the classifier metrics table |
| `paper_efficientnet_b3_best.pth` | EfficientNet-B3 classifier | Best checkpoint for the classifier metrics table |
| `paper_resnet50_best.pth` | ResNet-50 classifier | Best checkpoint for the classifier metrics table |
| `paper_vit_base_best.pth` | ViT-Base classifier | Reproduced ViT checkpoint used in the classifier comparison table |
| `paper_swin_base_best.pth` | Swin classifier | Best classifier and agentic-review backbone |
| `paper_yolo_best.pt` | YOLO detector | Best YOLO checkpoint used in the comparison workflow |

The corresponding paper metrics are in the published evaluation artifacts and report tables.

Use the `paper_*` links above when you want the exact checkpoint set tied to the reported results.

## Additional Checkpoints

These are kept because they are referenced elsewhere in the repo or report discussion:

| Link | Model / use | Notes |
| --- | --- | --- |
| `vit_base_100_epoch_best.pth` | ViT-Base 100-epoch run | Best 100-epoch ViT checkpoint discussed in the paper updates |
| `vit_cnn_sized_100_epoch_best.pth` | ViT-CNN-sized 100-epoch run | Additional 100-epoch ViT-family run |
| `standalone_agent_swin_best.pth` | Swin review backbone | Standalone path used by the review config |
| `standalone_yolo_best.pt` | YOLO review backbone | Standalone path used by the review config |

## Notes

- These files are symlinks, not copied checkpoints.
- If you move the underlying `experiments/` directories, these links must be regenerated.
- For exact paper-number verification, prefer the `paper_*` links above.
