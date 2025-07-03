# pdf_to_txt.py
import pdfplumber
from pathlib import Path
from config import DIR_COMPLETE_TXT      #  ← new import

def pdf_to_txt(pdf_path: str | Path) -> Path:
    pdf_path = Path(pdf_path)
    out_path = DIR_COMPLETE_TXT / pdf_path.with_suffix(".txt").name   #  ← changed

    with pdfplumber.open(pdf_path) as pdf, out_path.open("w", encoding="utf-8") as f:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                f.write(t + "\n")
    return out_path          # lets the caller know where it was saved
