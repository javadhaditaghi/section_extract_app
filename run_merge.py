# run_merge.py
"""
Standalone script to merge all datasheet CSV files into one consolidated file.

This script merges all individual thesis datasheet files from data/sectioned/datasheets/
into a single ALL_DATASHEETS.csv file.

Usage:
    python run_merge.py                    # Merge with default settings
    python run_merge.py --no-source       # Merge without source file info
    python run_merge.py --output custom.csv  # Custom output filename
"""

import argparse
from pathlib import Path
from src.preprocessing.merge_datasheets import merge_datasheets, print_merge_statistics
from src.utils.config import DIR_DATASHEETS


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Merge individual datasheet CSV files into one consolidated file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_merge.py                    # Basic merge
  python run_merge.py --no-source       # Without source file tracking
  python run_merge.py --output final.csv # Custom output name
  python run_merge.py --stats-only       # Just show existing stats
        """
    )

    parser.add_argument(
        '--output', '-o',
        default='ALL_DATASHEETS.csv',
        help='Output filename for merged datasheet (default: ALL_DATASHEETS.csv)'
    )

    parser.add_argument(
        '--no-source',
        action='store_true',
        help='Do not include source file information in merged data'
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=None,
        help='Input directory containing datasheet files (default: data/sectioned/datasheets/)'
    )

    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show statistics for existing merged file, do not merge'
    )

    args = parser.parse_args()

    # Handle stats-only mode
    if args.stats_only:
        input_dir = args.input_dir or DIR_DATASHEETS
        merged_file = input_dir / args.output

        if merged_file.exists():
            print_merge_statistics(merged_file)
        else:
            print(f"❌ Merged file not found: {merged_file}")
            print("   Run the merge first without --stats-only")
        return

    # Perform the merge
    print("🚀 Starting datasheet merge process...")
    print(f"📁 Input directory: {args.input_dir or DIR_DATASHEETS}")
    print(f"📄 Output filename: {args.output}")
    print(f"🔗 Include source info: {not args.no_source}")
    print("=" * 50)

    try:
        merged_file = merge_datasheets(
            input_dir=args.input_dir,
            output_filename=args.output,
            include_source_info=not args.no_source
        )

        if merged_file:
            print_merge_statistics(merged_file)
            print(f"\n🎉 Merge completed successfully!")
            print(f"   📄 Merged file: {merged_file}")
            print(f"   📊 Use 'python run_merge.py --stats-only' to view stats later")
        else:
            print("❌ Merge failed - no valid datasheet files found.")

    except Exception as e:
        print(f"❌ Error during merge: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())