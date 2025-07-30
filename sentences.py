# src/preprocessing/sentences.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import spacy
from src.utils.config import DIR_PROCESSED  # ✅ updated import path

"""Quick‑start
--------------
conda create -n nlp python=3.11
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
"""

MODEL_NAME = "en_core_sci_sm"

# ── Load SciSpaCy with sentencizer ──
try:
    nlp = spacy.load(MODEL_NAME, disable=["parser", "ner", "tagger"])
except (OSError, IOError) as e:
    raise OSError(
        f"SciSpaCy model '{MODEL_NAME}' is not installed.\n"
        "Install it with:\n"
        "pip install scispacy\n"
        f"pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{MODEL_NAME}-0.5.4.tar.gz"
    ) from e

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


def scispacy_sentences(text: str) -> List[str]:
    """Return a list of sentence strings segmented by SciSpaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def build_datasheet(
    name_part: str,
    abstract_text: str,
    intro_text: str,
    thesis_code: str
) -> Path:
    """Create one CSV in data/processed/ with a row per sentence."""
    rows: List[Dict[str, str | int]] = []

    def add(section: str, text: str) -> None:
        for idx, s in enumerate(scispacy_sentences(text), start=1):
            rows.append(
                {
                    "index": idx,
                    "thesis code": thesis_code,
                    "sentence": s,
                    "section": section,
                }
            )

    if abstract_text:
        add("Abstract", abstract_text)
    if intro_text:
        add("Introduction", intro_text)

    df = pd.DataFrame(rows)
    out_path = DIR_PROCESSED / f"{name_part}_datasheet.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📄 Datasheet saved to {out_path}")
    return out_path
