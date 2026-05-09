"""
Minimal .env loading helpers.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_local_env(env_path: str | Path = ".env") -> None:
    path = Path(env_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
