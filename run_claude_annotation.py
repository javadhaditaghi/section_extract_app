# run_claude_annotation.py
"""
Simple runner script for Claude-based metadiscourse annotation.

This script provides an easy way to run metadiscourse annotation on your merged dataset,
using Anthropic's Claude API for analysis.

Usage:
    python run_claude_annotation.py                    # Use default merged dataset
    python run_claude_annotation.py --sample 10       # Process only 10 sentences
    python run_claude_annotation.py --input custom.csv # Use custom input file
"""

import argparse
from pathlib import Path
from src.annotation.LLM.Claude.metadiscourse_annotator import MetadiscourseAnnotator
from src.utils.config import DIR_DATASHEETS, DIR_ANNOTATIONS_CLAUDE


def main():
    """Main function with simple command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run Claude-based metadiscourse annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_claude_annotation.py                    # Process all sentences in merged dataset
  python run_claude_annotation.py --sample 2        # Process only 2 sentences (for testing)
  python run_claude_annotation.py --sample 100      # Process 100 sentences
  python run_claude_annotation.py --input custom.csv # Use custom input file
  python run_claude_annotation.py --output results.csv # Custom output name
  python run_claude_annotation.py --model claude-3-opus-20240229 # Use Claude Opus
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
        default="claude-opus-4-20250514",
        choices=[
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219"
        ],
        help='Claude model to use (default: claude-opus-4-20250514)'
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

    print("🚀 Claude Metadiscourse Annotation")
    print("=" * 40)
    print(f"📄 Input file: {args.input}")
    print(f"🤖 Model: {args.model}")

    if args.sample:
        print(f"📊 Sample size: {args.sample} sentences")
    else:
        print("📊 Processing: All sentences")

    print(f"💾 Output directory: {DIR_ANNOTATIONS_CLAUDE}")
    print()

    try:
        # Initialize annotator
        print("🔧 Initializing Claude annotator...")
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
        print(f"   • Check data/annotations/LLM/Claude/ for all annotation files")
        print(f"   • Use the annotated data for further analysis")
        print(f"   • Compare results with GPT and DeepSeek annotations if available")

        return 0

    except Exception as e:
        print(f"\n❌ Annotation failed: {e}")
        print(f"💡 Make sure:")
        print(f"   • Claude API key is set in your .env file (CLAUDE_API_KEY)")
        print(f"   • You have sufficient API credits")
        print(f"   • The anthropic package is installed: pip install anthropic")
        print(f"   • The input file has a 'sentence' column")
        print(f"   • Your internet connection is stable")
        return 1


if __name__ == "__main__":
    exit(main())