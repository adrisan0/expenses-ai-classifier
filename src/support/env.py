"""Environment file helpers."""
from __future__ import annotations

import os
from pathlib import Path

from src.support.paths import ENV_FILE


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs from a local `.env` file."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        resolved_key = key.strip()
        resolved_value = value.strip().strip('"').strip("'")
        if resolved_key and resolved_key not in os.environ:
            os.environ[resolved_key] = resolved_value
