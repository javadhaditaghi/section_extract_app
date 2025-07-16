# sentences_scispacy.py (SciSpaCy version)
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, List

import pandas as pd
import spacy
from config import DIR_DATASHEETS

"""Quick‑start
--------------
conda create -n nlp python=3.11  # or use venv/poetry
pip install scispacy
# SMALL pipeline (fast, sentence splitting is identical across model sizes):
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
# MEDIUM or LARGE (adds vectors → slower but better for later NER/parse work):
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

After installing, run any script that imports this module; the pipeline will load automatically.
"""

MODEL_NAME = "en_core_sci_sm"  # change here if you installed *_md or *_lg

# ── Load a *lightweight* SciSpaCy pipeline and add only the sentencizer ──
try:
    # Disable heavy components we don't need for plain sentence segmentation
    nlp = spacy.load(MODEL_NAME, disable=["parser", "ner", "tagger"])
except (OSError, IOError) as e:
    raise OSError(
        f"SciSpaCy model '{MODEL_NAME}' is not installed.\n"
        "Install it with:\n"
        "pip install scispacy\n"
        "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/"
        f"{MODEL_NAME}-0.5.4.tar.gz\n"
        "(or replace with *_md / *_lg if you installed those)."
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
    """Create one CSV in datasheets/ with a row per sentence."""
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
    out_path = DIR_DATASHEETS / f"{name_part}_datasheet.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"📄 Datasheet saved to {out_path}")
    return out_path
