#!/usr/bin/env python3
"""
Runner script for cross-model annotation normalizer.
Execute this from the project root directory.

Usage:
    python run_normalizer.py

    # With custom settings:
    python run_normalizer.py --embedding-model all-mpnet-base-v2 --threshold 0.25 --lemmatize
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from postprocessing.cross_model_normalizer import CrossModelNormalizer, main as normalizer_main
    from utils.config import ensure_directories
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    print("Project structure should be:")
    print("  project_root/")
    print("  ├── src/")
    print("  │   ├── utils/config.py")
    print("  │   └── postprocessing/cross_model_normalizer.py")
    print("  ├── data/")
    print("  └── run_normalizer.py")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-Model Annotation Normalizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_normalizer.py                                    # Default settings
  python run_normalizer.py --threshold 0.25                   # Stricter clustering
  python run_normalizer.py --embedding-model all-mpnet-base-v2 # Better embeddings
  python run_normalizer.py --lemmatize                        # Enable lemmatization
  python run_normalizer.py --sample 1000                      # Process only 1000 samples
        """
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: all-MiniLM-L6-v2)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Clustering distance threshold (default: 0.3, similarity ≥ 0.7)"
    )

    parser.add_argument(
        "--lemmatize",
        action="store_true",
        help="Enable lemmatization of expressions"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only N samples (for testing)"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="normalized_annotations.csv",
        help="Output filename (default: normalized_annotations.csv)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_ROOT / "normalizer.log")
        ]
    )


def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'sentence_transformers',
        'sklearn',
        'pandas',
        'numpy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def main():
    """Main runner function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Ensure directories exist
    ensure_directories()

    print("🚀 Starting Cross-Model Annotation Normalizer")
    print("=" * 60)
    print(f"Embedding model: {args.embedding_model}")
    print(f"Clustering threshold: {args.threshold}")
    print(f"Lemmatization: {'Enabled' if args.lemmatize else 'Disabled'}")
    if args.sample:
        print(f"Sample size: {args.sample}")
    print("=" * 60)

    try:
        # Initialize normalizer with custom settings
        normalizer = CrossModelNormalizer(
            embedding_model_name=args.embedding_model,
            clustering_threshold=args.threshold,
            lemmatize=args.lemmatize
        )

        # Process annotations
        normalized_df = normalizer.process_all_annotations()

        # Apply sampling if requested
        if args.sample and len(normalized_df) > args.sample:
            print(f"\n📊 Sampling {args.sample} rows from {len(normalized_df)} total rows")
            normalized_df = normalized_df.sample(n=args.sample, random_state=42)

        # Save results
        normalizer.save_results(normalized_df, args.output_name)

        # Display summary
        print("\n✅ Normalization completed successfully!")
        print(f"📁 Output saved to: data/annotations/LLM/step1/{args.output_name}")
        print(f"📊 Total clusters: {len(normalized_df)}")
        print(f"🎯 Unique global_indices: {normalized_df['global_index'].nunique()}")
        print(f"📈 Average confidence: {normalized_df['avg_confidence'].mean():.3f}")

        # Show metadiscourse fraction distribution
        print("\n📋 Metadiscourse Fraction Distribution:")
        fraction_counts = normalized_df['metadiscourse_fraction'].value_counts().sort_index()
        for fraction, count in fraction_counts.items():
            models_count = int(fraction * 4)
            print(f"  {models_count} model(s) ({fraction:.2f}): {count} clusters")

        return normalized_df

    except Exception as e:
        print(f"\n❌ Error during normalization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    result_df = main()