#!/usr/bin/env python3
"""
Convenience entry point for the evaluation harness.
Delegates to evaluation/run_evaluation.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_evaluation import main

if __name__ == "__main__":
    main()
