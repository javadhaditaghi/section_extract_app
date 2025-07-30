# run_gpt_annotation.py
"""
Simple runner script for GPT-based metadiscourse annotation.

This script provides an easy way to run metadiscourse annotation on your merged dataset,
similar to your original annotator.py but integrated into the project ecosystem.

Usage:
    python run_gpt_annotation.py                    # Use default merged dataset
    python run_gpt_annotation.py --sample 10       # Process only 10 sentences
    python run_gpt_annotation.py --input custom.csv # Use custom input file
"""

import argparse
from pathlib import Path
from src.annotation.LLM.GPT.metadiscourse_annotator import MetadiscourseAnnotator
from src.utils.config import DIR_DATASHEETS, DIR_ANNOTATIONS_GPT


def main():
    """Main function with simple command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run GPT-based metadiscourse annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_gpt_annotation.py                    # Process all sentences in merged dataset
  python run_gpt_annotation.py --sample 2        # Process only 2 sentences (for testing)
  python run_gpt_annotation.py --sample 100      # Process 100 sentences
  python run_gpt_annotation.py --input custom.csv # Use custom input file
  python run_gpt_annotation.py --output results.xlsx # Custom output name
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=DIR_DATASHEETS / "ALL_DATASHEETS.csv",
        help='Input file path (default: data/sectioned/datasheets/ALL_DATASHEETS.csv)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (default: auto-generated with timestamp)'
    )

    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=None,
        help='Number of sentences to process (default: all)'
    )

    parser.add_argument(
        '--model', '-m',
        default="gpt-4",
        help='OpenAI model to use (default: gpt-4)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Quick test mode: process only 2 sentences'
    )

    args = parser.parse_args()

    # Handle test mode
    if args.test:
        args.sample = 2
        print("🧪 Running in test mode: processing only 2 sentences")

    # Validate input file
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        print("💡 Make sure you've run the merge process first:")
        print("   python run_merge.py")
        return 1

    print("🚀 GPT Metadiscourse Annotation")
    print("=" * 40)
    print(f"📄 Input file: {args.input}")
    print(f"🤖 Model: {args.model}")

    if args.sample:
        print(f"📊 Sample size: {args.sample} sentences")
    else:
        print("📊 Processing: All sentences")

    print(f"💾 Output directory: {DIR_ANNOTATIONS_GPT}")
    print()

    try:
        # Initialize annotator
        print("🔧 Initializing GPT annotator...")
        annotator = MetadiscourseAnnotator(model=args.model)

        # Run annotation
        result_path = annotator.annotate_file(
            input_path=args.input,
            output_path=args.output,
            sample_size=args.sample
        )

        print(f"\n🎉 Annotation completed successfully!")
        print(f"📄 Results saved to: {result_path}")

        # Show next steps
        print(f"\n💡 Next steps:")
        print(f"   • Review results in: {result_path}")
        print(f"   • Check data/annotations/LLM/GPT/ for all annotation files")
        print(f"   • Use the annotated data for further analysis")

        return 0

    except Exception as e:
        print(f"\n❌ Annotation failed: {e}")
        print(f"💡 Make sure:")
        print(f"   • OpenAI API key is set in your .env file")
        print(f"   • You have sufficient API credits")
        print(f"   • The input file has a 'sentence' column")
        return 1


if __name__ == "__main__":
    exit(main())