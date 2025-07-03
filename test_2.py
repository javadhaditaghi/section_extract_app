import re
from pathlib import Path

# ------------------------------------------------------------------
# 1)  Load text and normalise new‑lines
# ------------------------------------------------------------------
in_file  = Path("tesi_Lazzeretti_finale.txt")
name_part = in_file.name.split('.')[0]
raw_text = in_file.read_text(encoding="utf-8", errors="ignore")
text     = raw_text.replace("\r\n", "\n")          # CRLF → LF

# ------------------------------------------------------------------
# 2)  Define “stop” headings
#     – tweak these lists whenever you meet a new thesis layout
# ------------------------------------------------------------------
STOP_ABSTRACT = [
"Abstract (Italiano)",
    "RIASSUNTO",
    "Abstract (Italian)",
    "Abstract (in Italian)",
    "table of contents",
    "chapter 1",
    "introduction",
    "1. introduction",
    "1. Introduction",
    "acknowledgements",
    "keywords",
    "literature review",
    "background",


]

STOP_INTRO = [
    "literature review", "chapter 2", "2. literature review", "chapter ii", "Chapter 2", "Chapter II",
    "background", "methodology", "materials and methods",
    "research design", "theoretical framework", "2. Theoretical background"
]

# ------------------------------------------------------------------
# 3)  Build regex patterns
# ------------------------------------------------------------------
# pattern_abstract = rf"(?im)^abstract\s*\n+(.*?)(?=\n+(?:{'|'.join(map(re.escape, STOP_ABSTRACT))})|\Z)"

pattern_abstract = rf"""(?im)           # Flags: i=ignore case, m=multiline
^
abstract                         # literal "abstract"
(?:\s*\(                        # optional: whitespace + opening parenthesis
    (?:in\s+)?english           # "in " optional + "english"
\))?                            # closing parenthesis, whole group optional
\s*                             # optional whitespace
\n+                             # one or more newlines
(.*?)                           # non-greedy capture of everything until...
(?=                             # lookahead for...
    \n+                         # one or more newlines
    (?:{'|'.join(map(re.escape, STOP_ABSTRACT))})  # any stop words (escaped)
    | \Z                        # OR end of string
)
"""


pattern_intro = rf"""
    ^(?:chapter\s*1\s*[:.\-]?\s*)?     # optional “Chapter 1”
    (?:1\s*\.?\s*)?                    # optional “1.”
    introduction\s*[:\-]?\s*\n+        # the INTRODUCTION heading itself
    (.*?)                             # intro body
    (?=\n+(?:{'|'.join(map(re.escape, STOP_INTRO))})|\Z)  # stop at next section
"""


# ------------------------------------------------------------------
# 4)  Search and save
# ------------------------------------------------------------------
def extract(pattern, label, out_name):
    m = re.search(pattern, text,
                  flags=re.DOTALL | re.IGNORECASE | re.MULTILINE | re.VERBOSE)
    if m:
        body = m.group(1).strip()
        Path(out_name).write_text(body, encoding="utf-8")
        print(f"✓ {label} saved to {out_name}")
    else:
        print(f"⚠️  {label} not found.")


extract(pattern_abstract, "Abstract",     f"abstract_{name_part}.txt")
extract(pattern_intro,    "Introduction", f"introduction_{name_part}.txt")

