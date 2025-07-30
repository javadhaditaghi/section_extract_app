# src/preprocessing/pdf_to_txt.py

import pdfplumber
from pathlib import Path
from src.utils.config import DIR_COMPLETE_TXT


def pdf_to_txt(pdf_path: str | Path) -> Path:
    """
    Converts a PDF file to plain text using pdfplumber.

    Args:
        pdf_path (str | Path): Path to the input PDF file.

    Returns:
        Path: Path to the saved text file.
    """
    pdf_path = Path(pdf_path)

    # Ensure the input file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Create output path
    out_path = DIR_COMPLETE_TXT / pdf_path.with_suffix(".txt").name

    print(f"Converting PDF: {pdf_path}")
    print(f"Output will be saved to: {out_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            with out_path.open("w", encoding="utf-8") as f:
                total_pages = len(pdf.pages)
                print(f"Processing {total_pages} pages...")

                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        f.write(text + "\n")

                    # Progress indicator
                    if page_num % 10 == 0 or page_num == total_pages:
                        print(f"Processed {page_num}/{total_pages} pages")

        print(f"✓ PDF converted successfully: {out_path}")
        return out_path

    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        raise