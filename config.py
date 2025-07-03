# config.py   ← a tiny helper so every module sees the same folders
from pathlib import Path

BASE_DIR          = Path(__file__).parent        # project root
DIR_COMPLETE_TXT  = BASE_DIR / "complete_thesis"
DIR_ABSTRACTS     = BASE_DIR / "abstracts"
DIR_INTRODUCTIONS = BASE_DIR / "introductions"

for d in (DIR_COMPLETE_TXT, DIR_ABSTRACTS, DIR_INTRODUCTIONS):
    d.mkdir(exist_ok=True)
