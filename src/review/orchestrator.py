"""
End-to-end orchestration for Swin + YOLO + Claude review.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.review.builder import build_case_packet, encode_image_as_data_url, load_prompt_template, render_user_prompt
from src.review.gateway import CmuGatewayClient
from src.review.inference import SwinInferenceEngine, YoloInferenceEngine
from src.review.renderer import render_review_report
from src.review.schema import extract_json_object, validate_review_response


class ReviewOrchestrator:
    def __init__(
        self,
        review_config: dict[str, Any],
        prompt_path: str | Path,
        swin_engine: SwinInferenceEngine,
        yolo_engine: YoloInferenceEngine,
    ):
        self.review_config = review_config
        self.prompt_template = load_prompt_template(prompt_path)
        gateway_config = review_config["gateway"]
        self.client = CmuGatewayClient(
            base_url=gateway_config["base_url"],
            model=gateway_config["model"],
            timeout_seconds=int(gateway_config.get("timeout_seconds", 120)),
            temperature=float(gateway_config.get("temperature", 0.0)),
            max_tokens=int(gateway_config.get("max_tokens", 1200)),
        )
        self.max_retries = int(gateway_config.get("max_retries", 2))
        self.swin_engine = swin_engine
        self.yolo_engine = yolo_engine

    def _build_messages(self, image_path: str | Path, case_packet: dict[str, Any], retry_note: str | None = None) -> list[dict[str, Any]]:
        user_prompt = render_user_prompt(self.prompt_template, case_packet)
        if retry_note:
            user_prompt = (
                f"{user_prompt}\n\n"
                "The previous response failed validation. Return only one JSON object that matches the requested schema.\n"
                f"Validation error: {retry_note}\n"
            )
        return [
            {
                "role": "system",
                "content": (
                    "You are a radiology decision-support reviewer. "
                    "You must only use the provided image and structured model evidence. "
                    "You must respond with JSON only."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_as_data_url(image_path)},
                    },
                ],
            },
        ]

    def review_case(
        self,
        image_id: str,
        image_path: str | Path,
        cache_dir: str | Path | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        cache_path = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{image_id}_claude_review.json"
            if cache_path.exists() and not force_refresh:
                cached_result = json.loads(cache_path.read_text())
                validated = validate_review_response(cached_result["review"])
                cached_result["report_text"] = render_review_report(
                    validated,
                    case_packet=cached_result["case_packet"],
                )
                cache_path.write_text(json.dumps(cached_result, indent=2))
                return cached_result

        swin_result = self.swin_engine.predict(image_path)
        yolo_result = self.yolo_engine.predict(image_path)
        case_packet = build_case_packet(
            image_id=image_id,
            image_path=image_path,
            swin_probabilities=swin_result.probabilities,
            swin_predicted_labels=swin_result.predicted_labels,
            yolo_detections=yolo_result.detections,
            yolo_predicted_labels=yolo_result.predicted_labels,
            yolo_conf_threshold=self.yolo_engine.conf_threshold,
        )

        raw_text = ""
        last_error = ""
        for attempt in range(self.max_retries + 1):
            messages = self._build_messages(
                image_path=image_path,
                case_packet=case_packet,
                retry_note=last_error or None,
            )
            gateway_response = self.client.chat_completion_with_retries(
                messages=messages,
                max_retries=self.max_retries,
            )
            raw_text = gateway_response.text
            try:
                payload = extract_json_object(raw_text)
                validated = validate_review_response(payload)
                report_text = render_review_report(validated, case_packet=case_packet)
                result = {
                    "image_id": image_id,
                    "image_path": str(Path(image_path)),
                    "case_packet": case_packet,
                    "raw_response_text": raw_text,
                    "review": validated.to_dict(),
                    "report_text": report_text,
                    "gateway_response": gateway_response.raw_json,
                }
                if cache_path is not None:
                    cache_path.write_text(json.dumps(result, indent=2))
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt == self.max_retries:
                    raise

        raise RuntimeError("Claude review failed unexpectedly.")
