# config.py
from pathlib import Path

# Base directories
DIR_DATA = Path(__file__).resolve().parents[2] / "data"
DIR_PROCESSED = DIR_DATA / "processed"
DIR_COMPLETE_TXT = DIR_PROCESSED / "complete_txt"

# Ensure the output directory exists
DIR_COMPLETE_TXT.mkdir(parents=True, exist_ok=True)
