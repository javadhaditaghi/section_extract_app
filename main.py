# main.py - Example usage script
"""
Main script demonstrating how to use the PDF processing pipeline.

Usage:
    python main.py                    # Process all PDFs in data/raw/
    python main.py path/to/file.pdf   # Process a specific PDF
"""

import sys
from pathlib import Path
from src.preprocessing.pdf_to_txt import pdf_to_txt
from src.preprocessing.sections import extract_sections, process_all_texts
from src.preprocessing.merge_datasheets import merge_datasheets, print_merge_statistics
from src.utils.config import DIR_RAW, DIR_COMPLETE_TXT, DIR_DATASHEETS


def process_single_pdf(pdf_path: str | Path) -> None:
    """Process a single PDF through the complete pipeline."""
    pdf_path = Path(pdf_path)

    print(f"🚀 Starting pipeline for: {pdf_path.name}")
    print("=" * 60)

    try:
        # Step 1: Convert PDF to text
        print("📄 Step 1: Converting PDF to text...")
        txt_path = pdf_to_txt(pdf_path)

        # Step 2: Extract sections and create datasheet
        print("\n🔍 Step 2: Extracting sections...")
        extract_sections(txt_path)

        print(f"\n✅ Pipeline completed successfully for: {pdf_path.name}")

    except Exception as e:
        print(f"\n❌ Pipeline failed for {pdf_path.name}: {e}")
        raise


def process_all_pdfs() -> None:
    """Process all PDFs in the raw data directory."""
    pdf_files = list(DIR_RAW.glob("*.pdf"))

    if not pdf_files:
        print(f"⚠️  No PDF files found in: {DIR_RAW}")
        print("   Please place your PDF files in the data/raw/ directory.")
        return

    print(f"🚀 Found {len(pdf_files)} PDF files to process")
    print("=" * 60)

    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📋 Processing PDF {i}/{len(pdf_files)}: {pdf_file.name}")
        try:
            process_single_pdf(pdf_file)
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")
            continue

    print(f"\n🎉 Batch processing completed!")
    print(f"   Processed: {len(pdf_files)} PDF files")
    print(f"   Text files saved to: {DIR_COMPLETE_TXT}")
    print(f"   Check data/sectioned/ for extracted sections and datasheets")


def merge_all_datasheets() -> None:
    """Merge all individual datasheets into one consolidated file."""
    print("📊 Merging all datasheets...")
    merged_file = merge_datasheets()

    if merged_file:
        print_merge_statistics(merged_file)
        print(f"\n✅ All datasheets merged successfully!")
        print(f"   Merged file: {merged_file}")
    else:
        print("⚠️  No datasheets found to merge.")
        print("   Process some PDF/text files first to create datasheets.")


def process_existing_texts() -> None:
    """Process existing text files (skip PDF conversion)."""
    print("🔍 Processing existing text files...")
    process_all_texts()


def main():
    """Main function with command line argument handling."""
    if len(sys.argv) > 1:
        # Process specific PDF file
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            return
        if pdf_path.suffix.lower() != '.pdf':
            print(f"❌ File is not a PDF: {pdf_path}")
            return

        process_single_pdf(pdf_path)
    else:
        # Interactive menu
        print("📚 PDF Processing Pipeline")
        print("=" * 30)
        print("1. Process all PDFs in data/raw/")
        print("2. Process existing text files only")
        print("3. Merge all datasheets")
        print("4. Full pipeline (PDFs → Sections → Merge)")
        print("5. Exit")

        while True:
            try:
                choice = input("\nSelect option (1-5): ").strip()

                if choice == '1':
                    process_all_pdfs()
                    break
                elif choice == '2':
                    process_existing_texts()
                    break
                elif choice == '3':
                    merge_all_datasheets()
                    break
                elif choice == '4':
                    print("🚀 Running full pipeline...")
                    process_all_pdfs()
                    print("\n" + "=" * 50)
                    merge_all_datasheets()
                    break
                elif choice == '5':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break


if __name__ == "__main__":
    main()