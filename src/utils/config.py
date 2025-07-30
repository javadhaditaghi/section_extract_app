# src/utils/config.py
from pathlib import Path

# Base directories - going up 2 levels from config.py to reach project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIR_DATA = PROJECT_ROOT / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_PROCESSED = DIR_DATA / "processed"
DIR_COMPLETE_TXT = DIR_PROCESSED / "complete_txt"
DIR_SECTIONED = DIR_DATA / "sectioned"
DIR_ABSTRACTS = DIR_SECTIONED / "abstracts"
DIR_INTRODUCTIONS = DIR_SECTIONED / "introductions"
DIR_DATASHEETS = DIR_SECTIONED / "datasheets"


# Create all necessary directories
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DIR_RAW,
        DIR_COMPLETE_TXT,
        DIR_ABSTRACTS,
        DIR_INTRODUCTIONS,
        DIR_DATASHEETS
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ensured: {directory}")


# Ensure directories exist when config is imported
ensure_directories()