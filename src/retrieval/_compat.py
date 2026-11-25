"""Compatibility helpers for the original ``cli/lib`` implementation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI_DIR = ROOT / "cli"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))


def load_movies() -> list[dict]:
    from lib.constants import load_movies as _load_movies

    return _load_movies()
