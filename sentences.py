# sentences.py  (spaCy version)
# from __future__ import annotations
# from pathlib import Path
# from typing import Iterable, Dict, List
# import pandas as pd
# import spacy
# from config import DIR_DATASHEETS
#
# # ── Load a *lightweight* spaCy pipeline and add a sentencizer
# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
# if "sentencizer" not in nlp.pipe_names:
#     nlp.add_pipe("sentencizer")
#
# def spacy_sentences(text: str) -> Iterable[str]:
#     doc = nlp(text)
#     return [sent.text.strip() for sent in doc.sents]
#
# def build_datasheet(
#     name_part: str,
#     abstract_text: str,
#     intro_text: str,
# ) -> Path:
#     """Create one CSV in datasheets/ with a row per sentence."""
#     rows: List[Dict[str, str | int]] = []
#
#     def add(section: str, text: str) -> None:
#         for idx, s in enumerate(spacy_sentences(text), start=1):
#             rows.append(
#                 {
#                     "index": idx,
#                     "sentence": s,
#                     "section": section,
#                     "metadiscourse category": "",
#                     "metadiscourse feature": "",
#                 }
#             )
#
#     if abstract_text:
#         add("Abstract", abstract_text)
#     if intro_text:
#         add("Introduction", intro_text)
#
#     df = pd.DataFrame(rows)
#     out_path = DIR_DATASHEETS / f"{name_part}_datasheet.csv"
#     df.to_csv(out_path, index=False, encoding="utf-8")
#     print(f"📄 Datasheet saved to {out_path}")
#     return out_path


from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, List
import pandas as pd
import spacy
from config import DIR_DATASHEETS

# ── Load a lightweight pipeline
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])

# --------------------------------------------------------------------
# 1)  Custom rule: keep list‑item headings together
# --------------------------------------------------------------------
@spacy.language.Language.component("list_item_sentences")
def list_item_sentences(doc):
    """
    If we see a pattern like  <digit> '.' <CapitalWord>
    make the digit token the *start* of a sentence so
    the whole line stays together:  '1. Establishing the Study'
    """
    for i, token in enumerate(doc[:-2]):            # look ahead safely
        if (
            token.like_num                          # e.g. '1'
            and doc[i + 1].text == "."              # literal dot
            and doc[i + 2].is_title                 # next word capitalised
        ):
            token.is_sent_start = True              # sentence begins at the number
            doc[i + 1].is_sent_start = False        # don't split after '1.'
            doc[i + 2].is_sent_start = False        # keep the word in same sentence
    return doc

# Add the normal sentencizer first, then our tweak
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
if "list_item_sentences" not in nlp.pipe_names:
    nlp.add_pipe("list_item_sentences", after="sentencizer")

# --------------------------------------------------------------------
# 2)  Helpers exactly as before
# --------------------------------------------------------------------
def spacy_sentences(text: str) -> Iterable[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def build_datasheet(
    name_part: str,
    abstract_text: str,
    intro_text: str,
) -> Path:
    """Create one CSV in datasheets/ with a row per sentence."""
    rows: List[Dict[str, str | int]] = []

    def add(section: str, text: str) -> None:
        for idx, s in enumerate(spacy_sentences(text), start=1):
            rows.append(
                {
                    "index": idx,
                    "sentence": s,
                    "section": section,
                    "metadiscourse category": "",
                    "metadiscourse feature": "",
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

