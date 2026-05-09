# Project Report

This project aimed to build an **explainable chest X-ray decision-support pipeline** on the **VinBigData CXR dataset**. The core goal was not just to classify abnormalities, but to combine **image-level classification**, **localization**, and an **agentic reporting layer** into a workflow that could support radiologists with structured, evidence-grounded outputs.

We worked on a **15-label subset** of VinBigData available in the current repository. These labels were:

`Aortic enlargement`, `Atelectasis`, `Calcification`, `Cardiomegaly`, `Consolidation`, `ILD`, `Infiltration`, `Lung Opacity`, `Nodule/Mass`, `Other lesion`, `Pleural effusion`, `Pleural thickening`, `Pneumothorax`, `Pulmonary fibrosis`, and `No finding`.

The project evolved in five main stages.

First, we built and cleaned a **reproducible image-level classification pipeline**. The training path was simplified to a single multi-label setup using `BCEWithLogitsLoss`, `512x512` resized inputs, fixed train/validation splits, and consistent metric reporting. We retained both a **from-scratch CNN baseline** and several **transfer-learning models** initialized from ImageNet-family pretrained weights where applicable.

Second, we evaluated a range of classifiers to understand how much performance comes from model scale versus pretraining. The retained models included:
- `simple_cnn`
- `efficientnet_b3`
- `resnet50`
- `vit_base`
- `swin_base_patch4_window7_224`

The strongest model was consistently **Swin**, which gave the best macro AUC-ROC among the classification models. A simple from-scratch CNN was also surprisingly competitive, which is an important result because it shows that strong performance is not exclusively dependent on very large transformers.

## Classification Results

| Model | Type | Params | Training Setup | Best Macro AUC-ROC |
| --- | --- | ---: | --- | ---: |
| `simple_cnn` | from scratch CNN | 1.31M | 100 epochs | 0.9280 |
| `efficientnet_b3` | transfer learning | 11.49M | 10 epochs | 0.9504 |
| `resnet50` | transfer learning | 24.56M | 10 epochs | 0.9453 |
| `vit_base` | transformer | 86.84M | 10 epochs | 0.8417 |
| `swin_base_patch4_window7_224` | hierarchical transformer | 87.28M | 10 epochs | 0.9618 |

These reproduced runs were executed across 4 H100 GPUs. The Swin result remained the strongest overall classifier. The CNN baseline did not match Swin, but it clearly outperformed weaker baselines like the non-pretrained ViT path in this setup.

Third, we added a **localization branch** using **YOLOv8m**. This was motivated by the fact that chest X-ray analysis benefits not only from class prediction but also from spatial evidence. YOLO was trained from the raw VinBigData box annotations and evaluated in two ways:
- as a **detector**, using `mAP@0.5` and `mAP@0.5:0.95`
- as a **derived image-level predictor**, by converting detections into label presence/absence

Initial YOLO runs were weak but usable. After extending training and using all four GPUs, the detector improved meaningfully.

## YOLO Results

| YOLO Run | Setup | mAP@0.5 | mAP@0.5:0.95 | Exact-Match Acc | Macro Acc | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Initial full run | 10 epochs | 0.1723 | 0.0981 | 0.7470 | 0.9501 | 0.3188 |
| Improved 4-GPU run | 50 epochs | 0.2452 | 0.1295 | 0.7473 | 0.9567 | 0.5422 |

This showed that YOLO improved significantly with longer training and better resource use. However, YOLO still did **not** surpass Swin as the main image-level model. Its strongest contribution was as a **supporting localization model** rather than the primary predictor.

Fourth, we built an **agentic workflow** on top of these models. The idea was:

`CXR image -> Swin classifier -> YOLO localizer -> agentic AI reviewer -> structured report`

The agentic layer does not replace the vision models. Instead, it reads:
- full Swin probabilities
- predicted labels
- YOLO detections
- bounding boxes
- model agreement/disagreement

It then produces a **structured decision-support summary** with:
- supported findings
- uncertain findings
- confidence bands
- short impression
- review recommendation
- safety disclaimer

We also introduced a small set of **derived global buckets** such as:
- `No acute abnormality`
- `Cardiomediastinal abnormality`
- `Pleural abnormality`
- `Airspace or infectious-inflammatory pattern`
- `Chronic interstitial or fibrotic pattern`
- `Focal lesion or mass-like pattern`
- `Possible urgent thoracic abnormality`

These were not treated as ground-truth labels. Instead, they were used as interpretive summaries to make the reports more clinically readable.

Fifth, we created **presentation-quality PDFs** that compare:
- the original image
- ground-truth radiologist labels
- Swin predictions
- YOLO predictions
- YOLO bounding boxes
- structured agentic summary

These reports were deliberately standardized into a doctor-readable format rather than generic LLM prose dumps.

## Overall Interpretation

The strongest outcome of the project is not “we beat the literature with a new model.” The real contribution is a **multi-stage explainable decision-support pipeline**. Swin gives the best image-level classification. YOLO adds localization support. The agentic layer turns those outputs into a structured report that is easier to interpret than raw probabilities alone.

The key findings are:
- **Swin** is the best classifier in this repo.
- A **simple CNN** is a strong baseline and validates the importance of including non-transformer models.
- **YOLO** improves with more training and is useful for localization, but it is not yet strong enough to replace classification.
- The **agentic workflow** is valuable as an explainability and reporting layer, even when it does not always improve raw classification metrics.

## Current Best Numbers

| Component | Best Current Result |
| --- | ---: |
| Best classifier | `swin_base_patch4_window7_224` |
| Swin macro AUC-ROC | 0.9618 |
| Best CNN macro AUC-ROC | 0.9280 |
| Best YOLO mAP@0.5 | 0.2452 |
| Best YOLO mAP@0.5:0.95 | 0.1295 |
| YOLO derived macro accuracy | 0.9567 |
| YOLO derived macro F1 | 0.5422 |

## Final Conclusion

In summary, this project successfully moved from a basic classification benchmark toward a more clinically meaningful **decision-support system**. The final pipeline is not just a model; it is a workflow that combines classification, localization, and structured explanation. That makes it more defensible as a practical medical AI project and more useful for presentation or further research.
