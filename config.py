# config.py   ← a tiny helper so every module sees the same folders
from pathlib import Path

BASE_DIR          = Path(__file__).parent        # project root
DIR_COMPLETE_TXT  = BASE_DIR / "complete_thesis"
DIR_PDFS          = BASE_DIR / "pdf_theses"
DIR_ABSTRACTS     = BASE_DIR / "abstracts"
DIR_INTRODUCTIONS = BASE_DIR / "introductions"
DIR_DATASHEETS    = BASE_DIR / "datasheets"

for d in (DIR_PDFS, DIR_COMPLETE_TXT, DIR_ABSTRACTS, DIR_INTRODUCTIONS, DIR_DATASHEETS):
    d.mkdir(exist_ok=True)