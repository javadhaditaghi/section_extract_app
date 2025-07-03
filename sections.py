# sections.py
import re
from pathlib import Path
from config import DIR_ABSTRACTS, DIR_INTRODUCTIONS


# ------------------------------------------------------------------
# 1)  “Stop” headings   (tweak if we see new thesis layouts :/ )
# ------------------------------------------------------------------
STOP_ABSTRACT = [
    "Abstract (Italiano)", "RIASSUNTO", "Abstract (Italian)", "Abstract (in Italian)",
    "table of contents", "chapter 1", "introduction", "1. introduction",
    "1. Introduction", "acknowledgements", "keywords", "literature review", "background",
]

STOP_INTRO = [
    "literature review", "chapter 2", "2. literature review", "chapter ii", "Chapter 2",
    "Chapter II", "background", "methodology", "materials and methods",
    "research design", "theoretical framework", "2. Theoretical background",
]

# ------------------------------------------------------------------
# 2)  Regex patterns (VERBOSE flag keeps them readable)
# ------------------------------------------------------------------
PATTERN_ABSTRACT = rf"""
(?im)^abstract(?:\s*\((?:in\s+)?english\))?\s*\n+(.*?)          # body
(?=\n+(?:{'|'.join(map(re.escape, STOP_ABSTRACT))})|\Z)         # until next heading
"""

PATTERN_INTRO = rf"""
^(?:chapter\s*1\s*[:.\-]?\s*)?(?:1\s*\.?\s*)?introduction\s*[:\-]?\s*\n+(.*?)   # body
(?=\n+(?:{'|'.join(map(re.escape, STOP_INTRO))})|\Z)
"""


def save_match(pattern: str, label: str, text: str, out_path: Path) -> None:
    m = re.search(pattern, text,
                  flags=re.DOTALL | re.IGNORECASE | re.MULTILINE | re.VERBOSE)
    if m:
        body = m.group(1).strip()
        out_path.write_text(body, encoding="utf-8")
        print(f"✓ {label} saved to {out_path}")
    else:
        print(f"⚠️  {label} not found.")

def extract_sections(txt_path: str | Path) -> None:
    txt_path = Path(txt_path)
    name_part = txt_path.stem
    text = txt_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")

    save_match(
        PATTERN_ABSTRACT,
        "Abstract",
        text,
        DIR_ABSTRACTS / f"abstract_{name_part}.txt"
    )
    save_match(
        PATTERN_INTRO,
        "Introduction",
        text,
        DIR_INTRODUCTIONS / f"introduction_{name_part}.txt"
    )
