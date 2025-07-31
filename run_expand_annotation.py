# run_expand_annotation.py
"""
Simple runner for expanding metadiscourse annotation results.

This script provides an easy way to process and expand annotation results
from GPT, Claude, DeepSeek, or Gemini into structured CSV files.

Usage:
    python run_expand_annotation.py
"""

from src.postprocessing.expand_annotation import main

if __name__ == "__main__":
    main()