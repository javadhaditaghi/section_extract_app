# src/preprocessing/sentences.py (SciSpaCy version)
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import spacy
from src.utils.config import DIR_DATASHEETS

"""
SciSpaCy Setup Instructions:
---------------------------
1. Create virtual environment:
   conda create -n nlp python=3.11  # or use venv/poetry

2. Install scispacy:
   pip install scispacy

3. Install model (choose one):
   # SMALL pipeline (fast, recommended for sentence splitting):
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

   # MEDIUM (adds vectors, slower but better for NER/parsing):
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

   # LARGE (best quality, slowest):
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
"""

# Model configuration - change this if you installed a different model
MODEL_NAME = "en_core_sci_sm"  # or "en_core_sci_md" or "en_core_sci_lg"

# Initialize the NLP pipeline
nlp = None


def load_nlp_model():
    """Load the SciSpaCy model with error handling."""
    global nlp
    if nlp is not None:
        return nlp

    try:
        # Load model with only necessary components for sentence segmentation
        nlp = spacy.load(MODEL_NAME, disable=["parser", "ner", "tagger"])

        # Add sentencizer if not present
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        print(f"✓ SciSpaCy model '{MODEL_NAME}' loaded successfully")
        return nlp

    except (OSError, IOError) as e:
        error_msg = (
            f"❌ SciSpaCy model '{MODEL_NAME}' is not installed.\n"
            f"Please install it with:\n"
            f"pip install scispacy\n"
            f"pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/"
            f"{MODEL_NAME}-0.5.4.tar.gz\n"
            f"(or replace with *_md / *_lg if you prefer those models)"
        )
        print(error_msg)
        raise OSError(error_msg) from e


def scispacy_sentences(text: str) -> List[str]:
    """
    Return a list of sentence strings segmented by SciSpaCy.

    Args:
        text (str): Input text to segment into sentences

    Returns:
        List[str]: List of sentence strings
    """
    if not text.strip():
        return []

    nlp_model = load_nlp_model()
    doc = nlp_model(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    print(f"  → Segmented into {len(sentences)} sentences")
    return sentences


def build_datasheet(
        name_part: str,
        abstract_text: str,
        intro_text: str,
        thesis_code: str
) -> Path:
    """
    Create one CSV in datasheets/ with a row per sentence.

    Args:
        name_part (str): Base name for the output file
        abstract_text (str): Text of the abstract section
        intro_text (str): Text of the introduction section
        thesis_code (str): Derived thesis code

    Returns:
        Path: Path to the created CSV file
    """
    print(f"Building datasheet for: {name_part}")
    rows: List[Dict[str, str | int]] = []
    sentence_index = 1  # Global sentence index across all sections

    def add_section(section_name: str, text: str) -> None:
        """Add sentences from a section to the datasheet."""
        nonlocal sentence_index

        if not text.strip():
            print(f"  ⚠️  No text found for {section_name}")
            return

        print(f"  Processing {section_name}...")
        sentences = scispacy_sentences(text)

        for sentence in sentences:
            if sentence:  # Only add non-empty sentences
                rows.append({
                    "index": sentence_index,
                    "thesis code": thesis_code,
                    "sentence": sentence,
                    "section": section_name,
                })
                sentence_index += 1

        print(f"  ✓ Added {len(sentences)} sentences from {section_name}")

    # Process sections
    if abstract_text:
        add_section("Abstract", abstract_text)
    if intro_text:
        add_section("Introduction", intro_text)

    if not rows:
        print(f"  ⚠️  No sentences found for {name_part}")
        return None

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    out_path = DIR_DATASHEETS / f"{name_part}_datasheet.csv"

    try:
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✓ Datasheet saved: {out_path} ({len(rows)} sentences total)")
        return out_path
    except Exception as e:
        print(f"❌ Error saving datasheet: {e}")
        raise