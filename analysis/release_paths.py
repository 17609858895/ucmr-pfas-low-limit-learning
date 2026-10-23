"""Portable read/write locations for the released scientific code."""
from pathlib import Path
import os

REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get('PFAS_RAW_DIR', REPO/'inputs')).expanduser().resolve()
WORK = Path(os.environ.get('PFAS_WORK_DIR', REPO/'work')).expanduser().resolve()
DATA = WORK/'data'
RESULTS = WORK/'results'
DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
