# src/annotation/LLM/DeepSeek/metadiscourse_annotator.py
"""
DeepSeek-based Metadiscourse Annotation Module

This module provides functionality to analyze sentences for metadiscourse expressions
using DeepSeek's API, based on Hyland's (2005) metadiscourse framework.
"""

import requests
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
    DIR_ANNOTATIONS_DEEPSEEK,
    DIR_DATASHEETS,
    AnnotationConfig
)


class MetadiscourseAnnotator:
    """
    DeepSeek-based annotator for identifying metadiscourse expressions in academic text.

    Based on Hyland's (2005) metadiscourse model, this annotator identifies both
    interactive and interactional metadiscourse expressions in academic sentences.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize the MetadiscourseAnnotator.

        Args:
            api_key (str, optional): DeepSeek API key. If None, loads from environment.
            model (str, optional): DeepSeek model to use. Defaults to config setting.
        """
        # Load environment variables
        load_dotenv()

        # Set up DeepSeek API
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Please set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.api_url = "https://api.deepseek.com/chat/completions"
        self.model = model or AnnotationConfig.DEEPSEEK_MODEL
        self.temperature = AnnotationConfig.DEEPSEEK_TEMPERATURE
        self.max_retries = AnnotationConfig.DEEPSEEK_MAX_RETRIES
        self.retry_delay = AnnotationConfig.DEEPSEEK_RETRY_DELAY

        # System prompt for metadiscourse analysis
        self.system_prompt = self._get_system_prompt()

        print(f"✓ MetadiscourseAnnotator initialized with model: {self.model}")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for metadiscourse analysis."""
        prompt_path = Path("src/annotation/prompts/metadiscourse.txt")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    def analyze_sentence(self, sentence: str) -> str:
        """
        Analyze a single sentence for metadiscourse expressions.

        Args:
            sentence (str): The sentence to analyze.

        Returns:
            str: DeepSeek's analysis of metadiscourse expressions.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Sentence: {sentence}"}
            ],
            "temperature": self.temperature,
            "stream": False
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    response_data = response.json()
                    return response_data["choices"][0]["message"]["content"].strip()
                else:
                    raise Exception(f"API request failed with status {response.status_code}: {response.text}")

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
            output_path = DIR_ANNOTATIONS_DEEPSEEK / output_filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        try:
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
        print(f"🎯 Starting DeepSeek metadiscourse annotation workflow")
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

    parser = argparse.ArgumentParser(description="DeepSeek-based Metadiscourse Annotation")
    parser.add_argument("--input", "-i", type=Path, help="Input file path")
    parser.add_argument("--output", "-o", type=Path, help="Output file path")
    parser.add_argument("--sample", "-s", type=int, help="Sample size (number of sentences)")
    parser.add_argument("--model", "-m", default="deepseek-chat", help="DeepSeek model to use")

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