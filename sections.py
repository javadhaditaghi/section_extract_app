# # sections.py
# import re
# from pathlib import Path
# from config import DIR_ABSTRACTS, DIR_INTRODUCTIONS
# from sentences import build_datasheet   # already present from the first patch
#
#
#
# # ------------------------------------------------------------------
# # 1)  “Stop” headings   (tweak if we see new thesis layouts :/ )
# # ------------------------------------------------------------------
# STOP_ABSTRACT = [
#     "Abstract (Italiano)", "RIASSUNTO", "Abstract (Italian)", "Abstract (in Italian)",
#     "table of contents", "chapter 1", "introduction", "1. introduction",
#     "1. Introduction", "acknowledgements", "keywords", "literature review", "background",
# ]
#
# STOP_INTRO = [
#     "literature review", "chapter 2", "2. literature review", "chapter ii", "Chapter 2",
#     "Chapter II", "background", "methodology", "materials and methods",
#     "research design", "theoretical framework", "2. Theoretical background",
# ]
#
# # ------------------------------------------------------------------
# # 2)  Regex patterns (VERBOSE flag keeps them readable)
# # ------------------------------------------------------------------
# PATTERN_ABSTRACT = rf"""
# (?im)^abstract(?:\s*\((?:in\s+)?english\))?\s*\n+(.*?)          # body
# (?=\n+(?:{'|'.join(map(re.escape, STOP_ABSTRACT))})|\Z)         # until next heading
# """
#
# PATTERN_INTRO = rf"""
# ^(?:chapter\s*1\s*[:.\-]?\s*)?(?:1\s*\.?\s*)?introduction\s*[:\-]?\s*\n+(.*?)   # body
# (?=\n+(?:{'|'.join(map(re.escape, STOP_INTRO))})|\Z)
# """
#
#
# def save_match(pattern: str, label: str, text: str, out_path: Path) -> None:
#     m = re.search(pattern, text,
#                   flags=re.DOTALL | re.IGNORECASE | re.MULTILINE | re.VERBOSE)
#     if m:
#         body = m.group(1).strip()
#         out_path.write_text(body, encoding="utf-8")
#         print(f"✓ {label} saved to {out_path}")
#     else:
#         print(f"⚠️  {label} not found.")
#
# def extract_sections(txt_path: str | Path) -> None:
#     txt_path = Path(txt_path)
#     name_part = txt_path.stem
#     text = txt_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
#
#     save_match(
#         PATTERN_ABSTRACT,
#         "Abstract",
#         text,
#         DIR_ABSTRACTS / f"abstract_{name_part}.txt"
#     )
#     save_match(
#         PATTERN_INTRO,
#         "Introduction",
#         text,
#         DIR_INTRODUCTIONS / f"introduction_{name_part}.txt"
#     )


# sections.py
import re
from pathlib import Path
from config import DIR_ABSTRACTS, DIR_INTRODUCTIONS
from sentences import build_datasheet      # ← datasheet helper

# ------------------------------------------------------------------
# 1)  Helper
# ------------------------------------------------------------------
def derive_thesis_code(stem: str) -> str:
    """
    Convert a filename stem into a code like 'AL1', 'CS07', 'XX1', etc.
    • First two letters found → discipline (upper‑cased)
    • First run of digits      → index   (default '1' if no digits)
    """
    import re
    letters = re.sub(r"[^A-Za-z]", "", stem)[:2].upper() or "XX"
    digits  = re.search(r"\d+", stem)
    number  = digits.group(0) if digits else "1"
    return f"{letters}{number}"


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

# ------------------------------------------------------------------
# 3)  Helper to save a match AND return it
# ------------------------------------------------------------------
def save_match(pattern: str, label: str, text: str, out_path: Path) -> str:
    """
    Find the section, write it to disk, return its body (or "").
    """
    m = re.search(pattern, text,
                  flags=re.DOTALL | re.IGNORECASE | re.MULTILINE | re.VERBOSE)
    if m:
        body = m.group(1).strip()
        out_path.write_text(body, encoding="utf-8")
        print(f"✓ {label} saved to {out_path}")
        return body
    else:
        print(f"⚠️  {label} not found.")
        return ""

# ------------------------------------------------------------------
# 4)  Main entry point
# ------------------------------------------------------------------
def extract_sections(txt_path: str | Path) -> None:
    txt_path  = Path(txt_path)
    name_part = txt_path.stem
    thesis_code = derive_thesis_code(name_part)
    text      = txt_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")

    # -- save each section and capture its text
    abstract_text = save_match(
        PATTERN_ABSTRACT,
        "Abstract",
        text,
        DIR_ABSTRACTS / f"abstract_{name_part}.txt"
    )

    intro_text = save_match(
        PATTERN_INTRO,
        "Introduction",
        text,
        DIR_INTRODUCTIONS / f"introduction_{name_part}.txt"
    )

    # -- build the CSV datasheet (only if at least one section was found)
    if abstract_text or intro_text:
        build_datasheet(name_part, abstract_text, intro_text, thesis_code=thesis_code)
