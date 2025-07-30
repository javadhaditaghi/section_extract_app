# src/preprocessing/merge_datasheets.py
from pathlib import Path
import pandas as pd
from typing import List, Optional
from src.utils.config import DIR_DATASHEETS


def merge_datasheets(
        input_dir: Optional[Path] = None,
        output_filename: str = "ALL_DATASHEETS.csv",
        include_source_info: bool = True
) -> Path:
    """
    Merge all individual datasheet CSV files into a single consolidated file.

    Args:
        input_dir (Path, optional): Directory containing datasheet files.
                                   Defaults to DIR_DATASHEETS from config.
        output_filename (str): Name for the merged output file.
        include_source_info (bool): Whether to include source file information.

    Returns:
        Path: Path to the merged datasheet file.
    """
    if input_dir is None:
        input_dir = DIR_DATASHEETS

    print(f"🔍 Looking for datasheet files in: {input_dir}")

    # Find all datasheet CSV files
    datasheet_files = list(input_dir.glob("*_datasheet.csv"))

    if not datasheet_files:
        print(f"⚠️  No datasheet files found in: {input_dir}")
        print("   Make sure you have processed some text files first.")
        return None

    print(f"📋 Found {len(datasheet_files)} datasheet files to merge:")
    for file in datasheet_files:
        print(f"   → {file.name}")

    # Initialize merged dataframe
    merged_data = []
    total_sentences = 0
    file_stats = []

    # Process each datasheet file
    for i, csv_file in enumerate(datasheet_files, 1):
        try:
            print(f"📄 Processing file {i}/{len(datasheet_files)}: {csv_file.name}")

            # Read the CSV file
            df = pd.read_csv(csv_file, encoding='utf-8')

            # Validate required columns
            required_columns = ['index', 'thesis code', 'sentence', 'section']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"   ⚠️  Missing columns in {csv_file.name}: {missing_columns}")
                print(f"   Available columns: {list(df.columns)}")
                continue

            # Add source file information if requested
            if include_source_info:
                df['source_file'] = csv_file.name
                df['source_stem'] = csv_file.stem.replace('_datasheet', '')

            # Add processing order for tracking
            df['file_order'] = i

            # Collect statistics
            num_sentences = len(df)
            sections = df['section'].value_counts().to_dict()
            file_stats.append({
                'file': csv_file.name,
                'sentences': num_sentences,
                'sections': sections
            })

            # Add to merged data
            merged_data.append(df)
            total_sentences += num_sentences

            print(f"   ✓ Added {num_sentences} sentences")

        except Exception as e:
            print(f"   ❌ Error processing {csv_file.name}: {e}")
            continue

    if not merged_data:
        print("❌ No valid datasheet files could be processed.")
        return None

    # Concatenate all dataframes
    print(f"\n🔄 Merging {len(merged_data)} dataframes...")
    merged_df = pd.concat(merged_data, ignore_index=True)

    # Reindex globally (optional - creates a continuous index across all files)
    merged_df['global_index'] = range(1, len(merged_df) + 1)

    # Reorder columns for better readability
    base_columns = ['global_index', 'thesis code', 'sentence', 'section']
    if include_source_info:
        base_columns.extend(['source_file', 'source_stem'])
    base_columns.extend(['index', 'file_order'])  # Original index and file order at the end

    # Only include columns that exist in the dataframe
    final_columns = [col for col in base_columns if col in merged_df.columns]
    merged_df = merged_df[final_columns]

    # Save merged datasheet
    output_path = input_dir / output_filename
    try:
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Merged datasheet saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving merged file: {e}")
        raise

    # Print summary statistics
    print(f"\n📊 Merge Summary:")
    print(f"   Files processed: {len(merged_data)}/{len(datasheet_files)}")
    print(f"   Total sentences: {total_sentences:,}")
    print(f"   Output file: {output_path}")

    # Section breakdown
    section_counts = merged_df['section'].value_counts()
    print(f"\n📋 Section breakdown:")
    for section, count in section_counts.items():
        percentage = (count / total_sentences) * 100
        print(f"   {section}: {count:,} sentences ({percentage:.1f}%)")

    # Thesis code breakdown
    thesis_counts = merged_df['thesis code'].value_counts()
    print(f"\n🎓 Thesis breakdown ({len(thesis_counts)} unique):")
    for thesis, count in thesis_counts.head(10).items():  # Show top 10
        percentage = (count / total_sentences) * 100
        print(f"   {thesis}: {count:,} sentences ({percentage:.1f}%)")

    if len(thesis_counts) > 10:
        print(f"   ... and {len(thesis_counts) - 10} more")

    return output_path


def get_merge_statistics(merged_file: Path) -> dict:
    """
    Get detailed statistics about a merged datasheet file.

    Args:
        merged_file (Path): Path to the merged CSV file.

    Returns:
        dict: Dictionary containing various statistics.
    """
    if not merged_file.exists():
        raise FileNotFoundError(f"Merged file not found: {merged_file}")

    df = pd.read_csv(merged_file)

    stats = {
        'total_sentences': len(df),
        'unique_theses': df['thesis code'].nunique(),
        'sections': df['section'].value_counts().to_dict(),
        'thesis_distribution': df['thesis code'].value_counts().to_dict(),
        'avg_sentences_per_thesis': len(df) / df['thesis code'].nunique(),
        'file_info': {
            'size_mb': merged_file.stat().st_size / (1024 * 1024),
            'path': str(merged_file)
        }
    }

    if 'source_file' in df.columns:
        stats['source_files'] = df['source_file'].nunique()
        stats['files_distribution'] = df['source_file'].value_counts().to_dict()

    return stats


def print_merge_statistics(merged_file: Path) -> None:
    """Print detailed statistics about a merged datasheet file."""
    try:
        stats = get_merge_statistics(merged_file)

        print(f"\n📊 Detailed Statistics for: {merged_file.name}")
        print("=" * 50)
        print(f"Total sentences: {stats['total_sentences']:,}")
        print(f"Unique theses: {stats['unique_theses']}")
        print(f"Average sentences per thesis: {stats['avg_sentences_per_thesis']:.1f}")
        print(f"File size: {stats['file_info']['size_mb']:.2f} MB")

        if 'source_files' in stats:
            print(f"Source files: {stats['source_files']}")

        print(f"\nSection distribution:")
        for section, count in stats['sections'].items():
            percentage = (count / stats['total_sentences']) * 100
            print(f"  {section}: {count:,} ({percentage:.1f}%)")

    except Exception as e:
        print(f"❌ Error getting statistics: {e}")


if __name__ == "__main__":
    # Main execution
    print("🚀 Starting datasheet merge process...")
    merged_file = merge_datasheets()

    if merged_file:
        print_merge_statistics(merged_file)
        print(f"\n🎉 Merge process completed successfully!")
    else:
        print("❌ Merge process failed.")