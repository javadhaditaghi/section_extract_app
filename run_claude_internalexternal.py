#!/usr/bin/env python3
"""
Run script for Internal/External annotation using Claude.
This script should be run from the project root directory.

Usage:
    python run_internalexternal_annotation.py <input_file>
    python run_internalexternal_annotation.py <input_file> --output_dir <custom_output_dir>

Examples:
    python run_internalexternal_annotation.py sample_data.csv
    python run_internalexternal_annotation.py my_annotations.csv --output_dir data/custom_output/
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to Python path so we can import our modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the annotation script
try:
    from src.annotation.LLM.Claude.internalexternal_annotation import InternalExternalAnnotator
    from src.utils.config import DIR_ANNOTATIONS_CLAUDE, AnnotationConfig
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    print("Also ensure all required packages are installed: pip install pandas anthropic python-dotenv")
    sys.exit(1)


def list_available_files():
    """List available CSV files in the Claude annotations directory."""
    if not DIR_ANNOTATIONS_CLAUDE.exists():
        return []

    csv_files = list(DIR_ANNOTATIONS_CLAUDE.glob("*.csv"))
    # Filter out files that already have 'inex_' prefix to avoid processing already processed files
    csv_files = [f for f in csv_files if not f.name.startswith('inex_')]
    return sorted(csv_files)


def choose_file_interactively():
    """Allow user to choose from available files interactively."""
    available_files = list_available_files()

    if not available_files:
        print(f"❌ No CSV files found in {DIR_ANNOTATIONS_CLAUDE}")
        print("Please add some CSV files to process.")
        sys.exit(1)

    print(f"📁 Available files in {DIR_ANNOTATIONS_CLAUDE}:")
    print()
    for i, file_path in enumerate(available_files, 1):
        print(f"  {i}. {file_path.name}")

    print()
    while True:
        try:
            choice = input(f"Choose a file (1-{len(available_files)}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                print("Cancelled by user")
                sys.exit(0)

            file_index = int(choice) - 1
            if 0 <= file_index < len(available_files):
                return available_files[file_index].name
            else:
                print(f"Please enter a number between 1 and {len(available_files)}")

        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nCancelled by user")
            sys.exit(0)


def main():
    """Main function to run the internal/external annotation."""
    parser = argparse.ArgumentParser(
        description='Run Internal/External annotation using Claude API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_claude_internalexternal.py                          # Interactive file selection
  python run_claude_internalexternal.py sample_data.csv          # Process specific file
  python run_claude_internalexternal.py my_file.csv --output_dir data/custom_output/

The script will:
1. Read from data/annotations/LLM/Claude/<input_file>
2. Process each row with Claude API for Internal/External classification
3. Save results to data/annotations/LLM/Claude/inex_<input_file>

Make sure your .env file contains ANTHROPIC_API_KEY=your_api_key_here
        """
    )

    parser.add_argument(
        'input_file',
        nargs='?',  # Make it optional
        help='Input CSV file name (will be read from data/annotations/LLM/Claude/). If not provided, you can choose interactively.'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help=f'Custom output directory (default: {DIR_ANNOTATIONS_CLAUDE})'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be processed without actually running the annotation'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available files and exit'
    )

    args = parser.parse_args()

    # Handle --list option
    if args.list:
        available_files = list_available_files()
        if available_files:
            print(f"📁 Available files in {DIR_ANNOTATIONS_CLAUDE}:")
            for file_path in available_files:
                print(f"  - {file_path.name}")
        else:
            print(f"No CSV files found in {DIR_ANNOTATIONS_CLAUDE}")
        sys.exit(0)

    # Get input file name
    if args.input_file:
        input_filename = args.input_file
    else:
        # Interactive selection
        input_filename = choose_file_interactively()

    # Construct paths
    input_path = DIR_ANNOTATIONS_CLAUDE / input_filename
    output_dir = Path(args.output_dir) if args.output_dir else DIR_ANNOTATIONS_CLAUDE
    output_filename = f"inex_{Path(input_filename).stem}.csv"
    output_path = output_dir / output_filename

    # Validate input file exists
    if not input_path.exists():
        print(f"❌ Error: Input file not found at {input_path}")
        print(f"Available files in {DIR_ANNOTATIONS_CLAUDE}:")
        if DIR_ANNOTATIONS_CLAUDE.exists():
            csv_files = list(DIR_ANNOTATIONS_CLAUDE.glob("*.csv"))
            if csv_files:
                for file in csv_files:
                    print(f"  - {file.name}")
            else:
                print("  No CSV files found")
        else:
            print("  Directory does not exist")
        sys.exit(1)

    # Check if .env file exists
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        print(f"❌ Error: .env file not found at {env_path}")
        print("Please create a .env file with your ANTHROPIC_API_KEY")
        sys.exit(1)

    # Show configuration
    print("🔧 Configuration:")
    print(f"  Input file: {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Claude model: {AnnotationConfig.CLAUDE_MODEL}")
    print(f"  Temperature: {AnnotationConfig.CLAUDE_TEMPERATURE}")
    print(f"  Max tokens: {AnnotationConfig.CLAUDE_MAX_TOKENS}")
    print(f"  Request delay: {AnnotationConfig.REQUEST_DELAY}s")
    print()

    if args.dry_run:
        print("🔍 DRY RUN - No actual processing will be performed")

        # Show file info
        try:
            import pandas as pd
            df = pd.read_csv(input_path)
            print(f"📊 File contains {len(df)} rows")
            print("📋 Columns:", list(df.columns))

            if len(df) > 0:
                print("\n📝 Sample row:")
                sample_row = df.iloc[0]
                for col in ['index', 'thesis code', 'sentence', 'expression']:
                    if col in sample_row:
                        value = str(sample_row[col])[:100] + "..." if len(str(sample_row[col])) > 100 else sample_row[
                            col]
                        print(f"  {col}: {value}")

        except Exception as e:
            print(f"❌ Error reading file: {e}")

        print("\nTo run the actual annotation, remove the --dry_run flag")
        return

    # Confirm before processing
    try:
        import pandas as pd
        df = pd.read_csv(input_path)
        row_count = len(df)

        print(f"📊 About to process {row_count} rows")
        estimated_time = row_count * (AnnotationConfig.REQUEST_DELAY + 2)  # rough estimate
        print(f"⏱️  Estimated time: ~{estimated_time / 60:.1f} minutes")

        if row_count > 100:
            response = input("\n⚠️  This is a large file. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled by user")
                sys.exit(0)

    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        sys.exit(1)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the annotation
    print("\n🚀 Starting annotation process...")
    try:
        annotator = InternalExternalAnnotator()
        annotator.process_file(input_path, output_path)
        print(f"\n✅ Successfully completed! Results saved to {output_path}")

    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()