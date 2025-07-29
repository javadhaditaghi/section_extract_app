# Code/preprocessing/pdf_to_txt.py

import pdfplumber
from pathlib import Path
from src.utils.config import DIR_COMPLETE_TXT  # Updated import path

def pdf_to_txt(pdf_path: str | Path) -> Path:
    """
    Converts a PDF file to plain text using pdfplumber.

    Args:
        pdf_path (str | Path): Path to the input PDF file.

    Returns:
        Path: Path to the saved text file.
    """
    pdf_path = Path(pdf_path)
    out_path = DIR_COMPLETE_TXT / pdf_path.with_suffix(".txt").name

    with pdfplumber.open(pdf_path) as pdf, out_path.open("w", encoding="utf-8") as f:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                f.write(text + "\n")

    return out_path
