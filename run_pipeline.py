# run_pipeline.py
from pathlib import Path
import argparse
from config import DIR_PDFS
from pdf_to_txt import pdf_to_txt
from sections import extract_sections

def main():
    parser = argparse.ArgumentParser(description="Extract sections from a thesis PDF")
    parser.add_argument("pdf", help="Path to the thesis PDF")
    args = parser.parse_args()

    pdf_file = Path(args.pdf)


    # look inside DIR_PDFS automatically.
    if not pdf_file.is_absolute() and not pdf_file.exists():
        candidate = DIR_PDFS / pdf_file.name
        if candidate.exists():
            pdf_file = candidate
        else:
            raise FileNotFoundError(
                f"Could not find {pdf_file} in the current directory "
                f"or in {DIR_PDFS}"
            )


    print("🔍 Extracting full text…")
    txt_file = pdf_to_txt(pdf_file)           # saved in complete_thesis/
    print(f"✓ Text saved to {txt_file}")

    print("🔍 Extracting Abstract and Introduction…")
    extract_sections(txt_file)                # writes into abstracts/ and introductions/
    print("🏁 Done!")

if __name__ == "__main__":
    main()
