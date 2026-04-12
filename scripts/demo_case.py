#!/usr/bin/env python3
"""
Run the agentic workflow for a single chest X-ray image.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agent import AgentTriagePolicy, SwinTriageService


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Swin + triage agent on one image")
    parser.add_argument("image_path", type=str, help="Path to the chest X-ray image")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained Swin checkpoint")
    parser.add_argument("--image-size", type=int, default=512, help="Image size used for inference")
    args = parser.parse_args()

    service = SwinTriageService(
        checkpoint_path=args.checkpoint_path,
        image_size=args.image_size,
    )
    policy = AgentTriagePolicy()

    summary = service.summarize_image(args.image_path)
    decision = policy.decide(summary)

    payload = {
        "case_summary": summary.to_dict(),
        "agent_decision": decision.to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
