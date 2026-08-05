"""Compatibility wrapper that preserves `python main.py`."""
from __future__ import annotations

import sys

from src.app import main


if __name__ == "__main__":
    main(sys.argv[1:])
