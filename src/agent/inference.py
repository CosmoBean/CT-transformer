"""
Inference helpers for the Swin-based triage workflow.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ..data.dataset import CLASS_NAMES
from ..data.transforms import build_classification_transform
from ..models import SwinTransformerClassifier
from .schema import CaseSummary


def _sigmoid_tensor(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


class SwinTriageService:
    """
    Load a trained Swin model and produce case summaries for agent routing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 512,
        model_name: str = "swin_base_patch4_window7_224",
        num_classes: int = 15,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.image_size = image_size
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SwinTransformerClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=False,
            img_size=image_size,
        )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = build_classification_transform(image_size=image_size, is_train=False)
        self.class_names = CLASS_NAMES[:num_classes]
        self.no_finding_index = self.class_names.index("No finding")

    def summarize_image(self, image_path: str, image_id: str | None = None) -> CaseSummary:
        image_id = image_id or Path(image_path).stem
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as exc:
            return CaseSummary(
                image_id=image_id,
                image_path=str(image_path),
                class_probabilities={},
                top_findings=[],
                top_scores={},
                no_finding_probability=0.0,
                max_abnormal_probability=0.0,
                abnormal_probability_margin=0.0,
                moderate_findings_count=0,
                heuristic_binary_prediction="unknown",
                valid_image=False,
                error=str(exc),
            )

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = _sigmoid_tensor(logits)[0].detach().cpu()

        return self._build_summary_from_probabilities(
            image_id=image_id,
            image_path=str(image_path),
            probabilities=probabilities,
        )

    def summarize_batch(
        self,
        image_ids: list[str],
        image_paths: list[str],
        images: torch.Tensor,
    ) -> list[CaseSummary]:
        with torch.no_grad():
            logits = self.model(images.to(self.device))
            probabilities = _sigmoid_tensor(logits).detach().cpu()

        summaries = []
        for image_id, image_path, probs in zip(image_ids, image_paths, probabilities):
            summaries.append(
                self._build_summary_from_probabilities(
                    image_id=image_id,
                    image_path=image_path,
                    probabilities=probs,
                )
            )
        return summaries

    def _build_summary_from_probabilities(
        self,
        image_id: str,
        image_path: str,
        probabilities: torch.Tensor,
    ) -> CaseSummary:
        probs = probabilities.tolist()
        class_probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(self.class_names, probs)
        }

        abnormal_items = [
            (class_name, prob)
            for class_name, prob in class_probabilities.items()
            if class_name != "No finding"
        ]
        abnormal_items.sort(key=lambda item: item[1], reverse=True)

        top_abnormal = abnormal_items[0]
        second_abnormal = abnormal_items[1] if len(abnormal_items) > 1 else ("", 0.0)
        no_finding_probability = class_probabilities["No finding"]
        max_abnormal_probability = float(top_abnormal[1])
        abnormal_probability_margin = float(top_abnormal[1] - second_abnormal[1])
        moderate_findings_count = sum(prob >= 0.35 for _, prob in abnormal_items)
        heuristic_binary_prediction = (
            "normal"
            if no_finding_probability >= max_abnormal_probability
            else "abnormal"
        )

        top_findings = [
            class_name
            for class_name, prob in abnormal_items[:3]
            if prob >= 0.20
        ]
        if not top_findings and no_finding_probability >= max_abnormal_probability:
            top_findings = ["No finding"]

        top_scores = {
            class_name: float(prob)
            for class_name, prob in abnormal_items[:3]
        }
        top_scores["No finding"] = float(no_finding_probability)

        return CaseSummary(
            image_id=image_id,
            image_path=image_path,
            class_probabilities=class_probabilities,
            top_findings=top_findings,
            top_scores=top_scores,
            no_finding_probability=float(no_finding_probability),
            max_abnormal_probability=max_abnormal_probability,
            abnormal_probability_margin=abnormal_probability_margin,
            moderate_findings_count=int(moderate_findings_count),
            heuristic_binary_prediction=heuristic_binary_prediction,
            valid_image=True,
        )
