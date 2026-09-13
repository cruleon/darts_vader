"""Lancia la dashboard Streamlit.

Uso:
    uv run python scripts/run_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as stcli

APP_PATH = Path(__file__).resolve().parent.parent / "src" / "dartvision" / "dashboard" / "app.py"


def main() -> None:
    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
