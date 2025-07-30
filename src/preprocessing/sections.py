# src/preprocessing/sections.py
import re
from pathlib import Path
from src.utils.config import DIR_ABSTRACTS, DIR_INTRODUCTIONS
from src.preprocessing.sentences import build_datasheet


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def derive_thesis_code(stem: str) -> str:
    """
    Derive a thesis code from the filename stem.

    Args:
        stem (str): Filename without extension

    Returns:
        str: Thesis code in format like "AB123"
    """
    # Extract letters (first 2) and numbers
    letters = re.sub(r"[^A-Za-z]", "", stem)[:2].upper() or "XX"
    digits = re.search(r"\d+", stem)
    number = digits.group(0) if digits else "1"
    return f"{letters}{number}"


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""

    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
    text = text.strip()

    return text


# ------------------------------------------------------------------
# Stop Headings (patterns that mark the end of sections)
# ------------------------------------------------------------------
STOP_ABSTRACT = [
    "Abstract (Italiano)", "RIASSUNTO", "Abstract (Italian)", "Abstract (in Italian)",
    "table of contents", "chapter 1", "introduction", "1. introduction",
    "1. Introduction", "acknowledgements", "keywords", "literature review",
    "background", "Chapter 1", "INTRODUCTION", "Contents", "TABLE OF CONTENTS"
]

STOP_INTRO = [
    "literature review", "chapter 2", "2. literature review", "chapter ii",
    "Chapter 2", "Chapter II", "background", "methodology", "materials and methods",
    "research design", "theoretical framework", "2. Theoretical background",
    "2. Literature Review", "LITERATURE REVIEW", "2. LITERATURE REVIEW",
    "related work", "2. Related Work", "RELATED WORK"
]


# ------------------------------------------------------------------
# Regex Patterns for Section Extraction
# ------------------------------------------------------------------
def create_abstract_pattern() -> str:
    """Create regex pattern for abstract extraction."""
    stop_pattern = '|'.join(map(re.escape, STOP_ABSTRACT))
    return rf"""
    (?im)                                                          # Case insensitive, multiline
    ^abstract(?:\s*\((?:in\s+)?english\))?\s*\n+                  # Match "Abstract" with optional "(English)" or "(in English)"
    (.*?)                                                          # Capture the abstract content (group 1)
    (?=\n+(?:{stop_pattern})|\Z)                                  # Stop at any stop heading or end of text
    """


def create_intro_pattern() -> str:
    """Create regex pattern for introduction extraction."""
    stop_pattern = '|'.join(map(re.escape, STOP_INTRO))
    return rf"""
    (?im)                                                          # Case insensitive, multiline
    ^(?:chapter\s*1\s*[:.\-]?\s*)?                                # Optional "Chapter 1" prefix
    (?:1\s*\.?\s*)?                                               # Optional "1." prefix
    introduction\s*[:\-]?\s*\n+                                   # "Introduction" with optional colon/dash
    (.*?)                                                          # Capture the introduction content (group 1)
    (?=\n+(?:{stop_pattern})|\Z)                                  # Stop at any stop heading or end of text
    """


# ------------------------------------------------------------------
# Section Extraction and Saving
# ------------------------------------------------------------------
def save_section(pattern: str, label: str, text: str, out_path: Path) -> str:
    """
    Extract a section using regex pattern and save to file.

    Args:
        pattern (str): Regex pattern to match the section
        label (str): Human-readable label for the section
        text (str): Full text to search in
        out_path (Path): Path where to save the extracted section

    Returns:
        str: Extracted section text, or empty string if not found
    """
    try:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE | re.VERBOSE)

        if match:
            section_text = clean_text(match.group(1))

            if section_text:
                # Ensure output directory exists
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the section
                out_path.write_text(section_text, encoding="utf-8")
                print(f"✓ {label} extracted and saved to: {out_path}")
                print(f"  → Length: {len(section_text)} characters")
                return section_text
            else:
                print(f"⚠️  {label} found but appears to be empty after cleaning")
                return ""
        else:
            print(f"⚠️  {label} not found in text")
            return ""

    except Exception as e:
        print(f"❌ Error extracting {label}: {e}")
        return ""


# ------------------------------------------------------------------
# Main Extraction Function
# ------------------------------------------------------------------
def extract_sections(txt_path: str | Path) -> None:
    """
    Extract abstract and introduction sections from a text file.

    Args:
        txt_path (str | Path): Path to the input text file
    """
    txt_path = Path(txt_path)

    # Validate input
    if not txt_path.exists():
        print(f"❌ Text file not found: {txt_path}")
        return

    print(f"\n🔍 Extracting sections from: {txt_path.name}")

    # Derive thesis code and prepare output paths
    name_part = txt_path.stem
    thesis_code = derive_thesis_code(name_part)
    print(f"  → Thesis code: {thesis_code}")

    # Read the text file
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        text = text.replace("\r\n", "\n")  # Normalize line endings
        print(f"  → Text loaded: {len(text)} characters")
    except Exception as e:
        print(f"❌ Error reading text file: {e}")
        return

    if not text.strip():
        print(f"⚠️  Text file is empty: {txt_path}")
        return

    # Extract sections
    abstract_text = save_section(
        create_abstract_pattern(),
        "Abstract",
        text,
        DIR_ABSTRACTS / f"abstract_{name_part}.txt"
    )

    intro_text = save_section(
        create_intro_pattern(),
        "Introduction",
        text,
        DIR_INTRODUCTIONS / f"introduction_{name_part}.txt"
    )

    # Create datasheet if any sections were found
    if abstract_text or intro_text:
        print(f"\n📊 Creating datasheet...")
        try:
            datasheet_path = build_datasheet(name_part, abstract_text, intro_text, thesis_code=thesis_code)
            if datasheet_path:
                print(f"✅ Processing completed successfully for: {name_part}")
            else:
                print(f"⚠️  Datasheet creation failed for: {name_part}")
        except Exception as e:
            print(f"❌ Error creating datasheet: {e}")
    else:
        print(f"⚠️  No sections found in: {name_part}")
        print(f"     Abstract found: {'Yes' if abstract_text else 'No'}")
        print(f"     Introduction found: {'Yes' if intro_text else 'No'}")


# ------------------------------------------------------------------
# Batch Processing Function
# ------------------------------------------------------------------
def process_all_texts(txt_dir: Path = None) -> None:
    """
    Process all text files in a directory.

    Args:
        txt_dir (Path, optional): Directory containing text files.
                                 Defaults to DIR_COMPLETE_TXT from config.
    """
    if txt_dir is None:
        from src.utils.config import DIR_COMPLETE_TXT
        txt_dir = DIR_COMPLETE_TXT

    txt_files = list(txt_dir.glob("*.txt"))

    if not txt_files:
        print(f"⚠️  No text files found in: {txt_dir}")
        return

    print(f"🚀 Processing {len(txt_files)} text files from: {txt_dir}")

    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n{'=' * 50}")
        print(f"Processing file {i}/{len(txt_files)}")
        extract_sections(txt_file)

    print(f"\n🎉 Batch processing completed! Processed {len(txt_files)} files.")