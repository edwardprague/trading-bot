"""
strategy.py — Version Router
=============================
Thin router that reads STRATEGY_VERSION from environment variables
and delegates execution to the appropriate versioned strategy file.

Usage:
    STRATEGY_VERSION=v1 python3 strategy.py
"""

import os
import sys
import runpy
from pathlib import Path

version = os.environ.get("STRATEGY_VERSION", "v1")
base_dir = Path(__file__).parent
strategy_file = base_dir / f"strategy_{version}.py"

if not strategy_file.exists():
    print(f"ERROR: Strategy file not found: strategy_{version}.py", file=sys.stderr)
    sys.exit(1)

# Execute the versioned strategy as if it were __main__
runpy.run_path(str(strategy_file), run_name="__main__")
