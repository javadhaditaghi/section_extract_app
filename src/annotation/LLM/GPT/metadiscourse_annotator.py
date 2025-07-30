# src/annotation/LLM/GPT/metadiscourse_annotator.py
"""
GPT-based Metadiscourse Annotation Module

This module provides functionality to analyze sentences for metadiscourse expressions
using OpenAI's GPT models, based on Hyland's (2005) metadiscourse framework.
"""

from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os
import time
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from datetime import datetime

from src.utils.config import (
    DIR_ANNOTATIONS_GPT,
    DIR_DATASHEETS,
    AnnotationConfig
)


class MetadiscourseAnnotator:
    """
    GPT-based annotator for identifying metadiscourse expressions in academic text.

    Based on Hyland's (2005) metadiscourse model, this annotator identifies both
    interactive and interactional metadiscourse expressions in academic sentences.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize the MetadiscourseAnnotator.

        Args:
            api_key (str, optional): OpenAI API key. If None, loads from environment.
            model (str, optional): OpenAI model to use. Defaults to config setting.
        """
        # Load environment variables
        load_dotenv()

        # Set up OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model or AnnotationConfig.OPENAI_MODEL
        self.temperature = AnnotationConfig.OPENAI_TEMPERATURE
        self.max_retries = AnnotationConfig.OPENAI_MAX_RETRIES
        self.retry_delay = AnnotationConfig.OPENAI_RETRY_DELAY

        # System prompt for metadiscourse analysis
        self.system_prompt = self._get_system_prompt()

        print(f"✓ MetadiscourseAnnotator initialized with model: {self.model}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for metadiscourse analysis."""
        prompt_path = Path("src/annotation/prompts/metadiscourse.txt")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

#         return """You are an academic language expert assisting with the identification of potential metadiscourse expressions in academic writing, based on Hyland's (2005) model.
# Your task is to read the sentence carefully in its surrounding context and mark any expression that might serve a metadiscursive function, even if you're unsure about the exact category or if it turns out to be propositional. Err on the side of inclusion.
# Focus on lexical indicators and rhetorical intent, not surface form or part of speech alone. Use the function of the expression in context to decide whether it may contribute to the reader–writer relationship (interactional) or text organization (interactive).
#
# Instructions:
# Mark any expression in the sentence that could reasonably function as metadiscourse, based on its context and rhetorical purpose — not just the word list.
#
# Do not classify the expression yet (e.g., hedge, booster, etc.). That comes later.
#
# Skip routine, factual, or propositional uses of common words unless they serve a rhetorical function.
#
# Use the surrounding context to judge whether the expression:
#
# Organizes the discourse (interactive), or
#
# Positions the writer and reader (interactional).
#
# Avoid shallow keyword spotting. Judge by rhetorical intent, not just presence of a word.
# Use the keyword reference list below to help you detect common indicators — but never rely on keyword spotting alone.
#
#
# Output for each sentence should include:
#
#
# Each expression identified
#
# A confidence score (1–5)
#
# Optional note if unsure (e.g., "I'm not sure" or "borderline case")
#
# A brief justification based on its rhetorical role
#
# Use this list to guide your attention — these are common, not exhaustive:
#
#
# Interactional
# Hedges
# almost, apparently, appear to be, approximately, assume, believed, certain extent, certain level, certain amount, could, couldn't, doubt, essentially, estimate, frequently, generally, in general, indicate, largely, likely, mainly, may, maybe, might, mostly, often, perhaps, plausible, possible, possibly, presumably, probable, probably, relatively, seems, sometimes, somewhat, suggest, suspect, unlikely, uncertain, unclear, usually, would, wouldn't, little, not understood
#
# Boosters (Emphatics)
# actually, always, apparent, certain that, certainly, certainty, clearly, it is clear, conclusively, decidedly, definitely, demonstrate, determine, doubtless, essential, establish, in fact, the fact that, indeed, know, it is known that, must, never, no doubt, beyond doubt, obvious, obviously, of course, prove, show, sure, true, undoubtedly, well known, won't, even if, should, by far
#
# Attitude Markers
# admittedly, I agree, amazingly, appropriately, correctly, curiously, disappointing, disagree, even, fortunately, have to, hopefully, important, importantly, interest, interestingly, prefer, pleased, must, ought, remarkable, surprisingly, unfortunate, unfortunately, unusually, understandably
#
# Engagement Markers (Relational Markers)
# incidentally, by the way, consider, imagine, let, let us, let's, lets, our, recall, us, we, you, next, to begin, note, notice, your, one's, think about
#
# Self-Mention (Person Markers)
# I, we, me, my, our, mine
#
#
# Interactive
# Transitions (Logical Connectives)
# and, but, therefore, thereby, so as to, in addition, similarly, equally, likewise, moreover, furthermore, in contrast, by contrast, as a result, the result is, result in, since, because, consequently, as a consequence, accordingly, on the other hand, on the contrary, however, also, yet, or
#
# Frame Markers (Announce Goals)
# my purpose, the aim, I intend, I seek, I wish, I argue, I propose, I suggest, I discuss, I would like to, I will focus on, we will focus on, I will emphasise, we will emphasise, my goal is, in this section, in this chapter, here I do this, here I will
#
# Frame Markers (Label Stages)
# to conclude, in conclusion, to sum up, in sum, summarise, summarize, overall, on the whole, all in all, so far, thus far, to repeat
#
# Frame Markers (Sequencing)
# to start with, first, firstly, second, secondly, third, thirdly, fourth, fourthly, fifthly, next, last, two, three, four, five
#
# Output Format (for each sentence):
# Provide the output as a list of identified expressions in the following structure for each entry:
# {
#     "expression": "<highlighted metadiscourse expression>",
#     "confidence": <1–5>,
#     "note": "<optional note if unsure, else leave blank>",
#     "justification": "<brief reason based on rhetorical intent>"
#   },
#   {
#     "expression_2": "<2nd highlighted metadiscourse expression>",
#     "confidence": <1–5>,
#     "note": "<optional note if unsure, else leave blank>",
#     "justification": "<brief reason based on rhetorical intent>"
#   },
#
# """

    def analyze_sentence(self, sentence: str) -> str:
        """
        Analyze a single sentence for metadiscourse expressions.

        Args:
            sentence (str): The sentence to analyze.

        Returns:
            str: GPT's analysis of metadiscourse expressions.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Sentence: {sentence}"}
                    ],
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                print(f"  ⚠️  Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    return f"ERROR after {self.max_retries} attempts: {e}"

    def load_data(self, file_path: Union[str, Path], sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from CSV or Excel file.

        Args:
            file_path (Union[str, Path]): Path to the input file.
            sample_size (int, optional): Number of rows to sample. None for all.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        print(f"📄 Loading data from: {file_path}")

        # Load based on file extension
        if file_path.suffix.lower() == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Validate required column
        if 'sentence' not in df.columns:
            raise ValueError(f"Required column 'sentence' not found. Available columns: {list(df.columns)}")

        # Sample data if requested
        original_size = len(df)
        if sample_size and sample_size < len(df):
            df = df.head(sample_size)
            print(f"📊 Using sample of {len(df)} rows (out of {original_size})")
        else:
            print(f"📊 Processing all {len(df)} rows")

        return df

    def annotate_dataframe(
            self,
            df: pd.DataFrame,
            batch_size: int = None,
            save_intermediate: bool = None,
            output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Annotate all sentences in a dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing sentences to annotate.
            batch_size (int, optional): Process in batches. Defaults to config setting.
            save_intermediate (bool, optional): Save progress periodically.
            output_path (Path, optional): Path for intermediate saves.

        Returns:
            pd.DataFrame: Annotated dataframe.
        """
        batch_size = batch_size or AnnotationConfig.DEFAULT_BATCH_SIZE
        save_intermediate = save_intermediate if save_intermediate is not None else AnnotationConfig.SAVE_INTERMEDIATE

        # Create a copy to avoid modifying original
        df_annotated = df.copy()

        # Add annotation column if it doesn't exist
        if "Metadiscourse Analysis" not in df_annotated.columns:
            df_annotated["Metadiscourse Analysis"] = ""

        # Add metadata columns
        df_annotated["annotation_timestamp"] = ""
        df_annotated["annotation_model"] = self.model

        print(f"🚀 Starting annotation of {len(df_annotated)} sentences...")
        print(f"   Model: {self.model}")
        print(f"   Batch size: {batch_size}")
        print(f"   Save intermediate: {save_intermediate}")

        # Process sentences
        for i, idx in enumerate(tqdm(df_annotated.index, desc="Analyzing Sentences")):
            sentence = df_annotated.at[idx, "sentence"]

            # Skip if already annotated (for resuming interrupted runs)
            if df_annotated.at[idx, "Metadiscourse Analysis"]:
                continue

            # Analyze sentence
            result = self.analyze_sentence(sentence)
            df_annotated.at[idx, "Metadiscourse Analysis"] = result
            df_annotated.at[idx, "annotation_timestamp"] = datetime.now().isoformat()

            # Save intermediate results
            if save_intermediate and output_path and (i + 1) % batch_size == 0:
                self._save_intermediate(df_annotated, output_path, i + 1)

        print("✅ Annotation completed!")
        return df_annotated

    def _save_intermediate(self, df: pd.DataFrame, output_path: Path, processed_count: int):
        """Save intermediate results."""
        intermediate_path = output_path.parent / f"{output_path.stem}_intermediate_{processed_count}{output_path.suffix}"
        try:
            intermediate_path = intermediate_path.with_suffix('.csv')
            df.to_csv(intermediate_path, index=False)

            print(f"  💾 Intermediate results saved: {intermediate_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to save intermediate results: {e}")

    def save_results(self, df: pd.DataFrame, output_path: Optional[Path] = None,
                     filename: Optional[str] = None) -> Path:
        """
        Save annotated results to file.

        Args:
            df (pd.DataFrame): Annotated dataframe to save.
            output_path (Path, optional): Specific output path.
            filename (str, optional): Custom filename.

        Returns:
            Path: Path where results were saved.
        """
        if output_path is None:
            # Generate default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if filename:
                output_filename = filename
            else:
                output_filename = f"metadiscourse_analysis_{timestamp}.{AnnotationConfig.OUTPUT_FORMAT}"
            output_path = DIR_ANNOTATIONS_GPT / output_filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        try:
            # if AnnotationConfig.OUTPUT_FORMAT == "xlsx" or output_path.suffix.lower() == '.xlsx':
            #     df.to_excel(output_path, index=False)
            # else:
            #     df.to_csv(output_path, index=False)
            df.to_csv(output_path.with_suffix('.csv'), index=False)

            print(f"✅ Results saved to: {output_path}")
            print(f"   Rows: {len(df)}")
            print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")

            return output_path

        except Exception as e:
            print(f"❌ Error saving results: {e}")
            raise

    def annotate_file(
            self,
            input_path: Union[str, Path],
            output_path: Optional[Path] = None,
            sample_size: Optional[int] = None,
            **kwargs
    ) -> Path:
        """
        Complete annotation workflow for a file.

        Args:
            input_path (Union[str, Path]): Path to input file.
            output_path (Path, optional): Path for output file.
            sample_size (int, optional): Number of sentences to process.
            **kwargs: Additional arguments for annotate_dataframe.

        Returns:
            Path: Path to saved results.
        """
        print(f"🎯 Starting metadiscourse annotation workflow")
        print("=" * 50)

        # Load data
        df = self.load_data(input_path, sample_size)

        # Annotate
        df_annotated = self.annotate_dataframe(df, output_path=output_path, **kwargs)

        # Save results
        result_path = self.save_results(df_annotated, output_path)

        print(f"\n🎉 Annotation workflow completed!")
        print(f"   Input: {input_path}")
        print(f"   Output: {result_path}")

        return result_path


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="GPT-based Metadiscourse Annotation")
    parser.add_argument("--input", "-i", type=Path, help="Input file path")
    parser.add_argument("--output", "-o", type=Path, help="Output file path")
    parser.add_argument("--sample", "-s", type=int, help="Sample size (number of sentences)")
    parser.add_argument("--model", "-m", default="gpt-4", help="OpenAI model to use")

    args = parser.parse_args()

    # Default input file
    input_file = args.input or DIR_DATASHEETS / "ALL_DATASHEETS.csv"

    # Initialize annotator
    annotator = MetadiscourseAnnotator(model=args.model)

    # Run annotation
    annotator.annotate_file(
        input_path=input_file,
        output_path=args.output,
        sample_size=args.sample
    )


if __name__ == "__main__":
    main()