# import pandas as pd
# import os
# import anthropic
# from dotenv import load_dotenv
# import argparse
# import sys
# from pathlib import Path
# import time
# import json
#
# # Import your existing config
# from src.utils.config import (
#     DIR_ANNOTATIONS_CLAUDE,
#     DIR_SRC,
#     AnnotationConfig
# )
#
#
# class InternalExternalAnnotator:
#     def __init__(self):
#         """Initialize the annotator with Claude API client."""
#         load_dotenv()
#         api_key = os.getenv('CLAUDE_API_KEY')
#         if not api_key:
#             raise ValueError("ANTHROPIC_API_KEY not found in .env file")
#
#         self.client = anthropic.Anthropic(api_key=api_key)
#         self.prompt_template = self.load_prompt()
#
#     def load_prompt(self):
#         """Load the prompt template from the prompts file."""
#         prompt_path = DIR_SRC / "annotation" / "prompts" / "internalexternal.txt"
#         if not prompt_path.exists():
#             raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
#
#         with open(prompt_path, 'r', encoding='utf-8') as f:
#             return f.read().strip()
#
#     def get_internal_external_annotation(self, row):
#         """
#         Get internal/external annotation for a single row using Claude API.
#
#         Args:
#             row: Pandas Series containing the row data
#
#         Returns:
#             str: "Internal" or "External" classification
#         """
#         # Prepare the context for the prompt
#         context = f"""
# Thesis Code: {row.get('thesis code', 'N/A')}
# Sentence: {row.get('sentence', 'N/A')}
# Section: {row.get('section', 'N/A')}
# Expression: {row.get('expression', 'N/A')}
# Confidence: {row.get('confidence', 'N/A')}
# Justification: {row.get('justification', 'N/A')}
# Note: {row.get('note', 'N/A')}
# """
#
#         # Combine prompt template with context
#         full_prompt = f"{self.prompt_template} \n\n{context}"
#
#         try:
#             response = self.client.messages.create(
#                 model=AnnotationConfig.CLAUDE_MODEL,
#                 max_tokens=AnnotationConfig.CLAUDE_MAX_TOKENS,
#                 temperature=AnnotationConfig.CLAUDE_TEMPERATURE,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": full_prompt
#                     }
#                 ]
#             )
#
#             return response.content[0].text.strip()
#             # # Extract the classification from the response
#             # response_text = response.content[0].text.strip()
#             #
#             # # Look for "Internal" or "External" in the response
#             # response_lower = response_text.lower()
#             # if "internal" in response_lower and "external" not in response_lower:
#             #     return "Internal"
#             # elif "external" in response_lower and "internal" not in response_lower:
#             #     return "External"
#             # elif "internal" in response_lower and "external" in response_lower:
#             #     # If both are mentioned, look for the first one or try to determine from context
#             #     if response_lower.find("internal") < response_lower.find("external"):
#             #         return "Internal"
#             #     else:
#             #         return "External"
#             # else:
#             #     print(f"Warning: Could not determine classification from response: {response_text[:100]}...")
#             #     return "Unknown"
#
#         except Exception as e:
#             print(f"Error processing row {row.get('index', 'unknown')}: {str(e)}")
#             return "Error"
#
#     def process_file(self, input_file_path, output_file_path):
#         """
#         Process the input file and create output with Internal_external column.
#
#         Args:
#             input_file_path: Path to input CSV file
#             output_file_path: Path to output CSV file
#         """
#         print(f"Loading data from {input_file_path}...")
#
#         # Read the input file
#         try:
#             df = pd.read_csv(input_file_path)
#         except Exception as e:
#             raise ValueError(f"Error reading input file: {str(e)}")
#
#         print(f"Loaded {len(df)} rows")
#
#         # Initialize the new column
#         df['Internal_external'] = ""
#
#         # Process each row
#         for idx, row in df.iterrows():
#             print(f"Processing row {idx + 1}/{len(df)} (index: {row.get('index', 'N/A')})")
#
#             classification = self.get_internal_external_annotation(row)
#             df.at[idx, 'Internal_external'] = classification
#
#             # Small delay to avoid rate limiting
#             time.sleep(AnnotationConfig.REQUEST_DELAY)
#
#         # Save the output file
#         print(f"Saving results to {output_file_path}...")
#         df.to_csv(output_file_path, index=False)
#         print(f"Processing complete! Results saved to {output_file_path}")
#
#         # Print summary
#         classification_counts = df['Internal_external'].value_counts()
#         print("\nClassification Summary:")
#         for classification, count in classification_counts.items():
#             print(f"  {classification}: {count}")
#
#
# def main():
#     """Main function to handle command line arguments and run the annotation."""
#     parser = argparse.ArgumentParser(description='Annotate data with Internal/External classification using Claude')
#     parser.add_argument('input_file', help='Input CSV file path (relative to Claude annotations directory)')
#     parser.add_argument('--output_dir', default=None,
#                         help=f'Output directory (default: {DIR_ANNOTATIONS_CLAUDE})')
#
#     args = parser.parse_args()
#
#     # Construct full paths using config
#     input_path = DIR_ANNOTATIONS_CLAUDE / args.input_file
#
#     # Generate output filename
#     input_stem = Path(args.input_file).stem
#     output_filename = f"inex_{input_stem}.csv"
#
#     # Use config directory if no output directory specified
#     output_dir = Path(args.output_dir) if args.output_dir else DIR_ANNOTATIONS_CLAUDE
#     output_path = output_dir / output_filename
#
#     # Validate input file exists
#     if not input_path.exists():
#         print(f"Error: Input file not found at {input_path}")
#         sys.exit(1)
#
#     # Create output directory if it doesn't exist
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#
#     try:
#         # Initialize annotator and process file
#         annotator = InternalExternalAnnotator()
#         annotator.process_file(input_path, output_path)
#
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         sys.exit(1)
#
#
# if __name__ == "__main__":
#     main()


import pandas as pd
import os
import anthropic
from dotenv import load_dotenv
import argparse
import sys
from pathlib import Path
import time
import json
import re

# Import your existing config
from src.utils.config import (
    DIR_ANNOTATIONS_CLAUDE,
    DIR_SRC,
    AnnotationConfig
)


class InternalExternalAnnotator:
    def __init__(self):
        """Initialize the annotator with Claude API client."""
        load_dotenv()
        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env file")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.prompt_template = self.load_prompt()

    def load_prompt(self):
        """Load the prompt template from the prompts file."""
        prompt_path = DIR_SRC / "annotation" / "prompts" / "internalexternal.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found at {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def extract_json_from_response(self, response_text):
        """Extract JSON data from various possible response formats."""
        try:
            # Try to find JSON in common patterns
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Clean up potential trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)

            # If not found as object, try array pattern
            array_pattern = r'\[.*\]'
            match = re.search(array_pattern, response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Clean up potential trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                data = json.loads(json_str)
                if isinstance(data, list) and len(data) > 0:
                    return data[0]  # Return first item if it's an array

            # If no JSON found, try to parse as simple response
            response_lower = response_text.lower()
            if "internal" in response_lower or "external" in response_lower:
                return {
                    "Internal/External": "Internal" if "internal" in response_lower else "External",
                    "confidence": 3,  # Default confidence
                    "note": "No note provided",
                    "justification": response_text.strip()
                }

            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {str(e)}")
            return None

    def get_internal_external_annotation(self, row):
        """
        Get internal/external annotation for a single row using Claude API.

        Args:
            row: Pandas Series containing the row data

        Returns:
            dict: Dictionary containing classification, justification, note, and confidence
        """
        # Prepare the context for the prompt
        context = f"""
Thesis Code: {row.get('thesis code', 'N/A')}
Sentence: {row.get('sentence', 'N/A')}
Section: {row.get('section', 'N/A')}
Expression: {row.get('expression', 'N/A')}
Confidence: {row.get('confidence', 'N/A')}
Justification: {row.get('justification', 'N/A')}
Note: {row.get('note', 'N/A')}
"""

        # Combine prompt template with context
        full_prompt = f"{self.prompt_template} \n\n{context}"

        try:
            response = self.client.messages.create(
                model=AnnotationConfig.CLAUDE_MODEL,
                max_tokens=AnnotationConfig.CLAUDE_MAX_TOKENS,
                temperature=AnnotationConfig.CLAUDE_TEMPERATURE,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ]
            )

            response_text = response.content[0].text.strip()
            json_data = self.extract_json_from_response(response_text)

            if not json_data:
                print(f"Warning: Could not extract JSON data from response: {response_text[:100]}...")
                return {
                    "Internal/External": "Unknown",
                    "confidence": 0,
                    "note": "Could not parse response",
                    "justification": response_text
                }

            return json_data

        except Exception as e:
            print(f"Error processing row {row.get('index', 'unknown')}: {str(e)}")
            return {
                "Internal/External": "Error",
                "confidence": 0,
                "note": str(e),
                "justification": "Error occurred during processing"
            }

    def process_file(self, input_file_path, output_file_path):
        """
        Process the input file and create output with Internal_external column.

        Args:
            input_file_path: Path to input CSV file
            output_file_path: Path to output CSV file
        """
        print(f"Loading data from {input_file_path}...")

        # Read the input file
        try:
            df = pd.read_csv(input_file_path)
        except Exception as e:
            raise ValueError(f"Error reading input file: {str(e)}")

        print(f"Loaded {len(df)} rows")

        # Initialize the new columns
        df['internal vs external'] = ""
        df['inex_justification'] = ""
        df['inex_note'] = ""
        df['inex_confidence'] = ""

        # Process each row
        for idx, row in df.iterrows():
            print(f"Processing row {idx + 1}/{len(df)} (index: {row.get('index', 'N/A')})")

            result = self.get_internal_external_annotation(row)

            # Update the DataFrame with extracted values
            df.at[idx, 'internal vs external'] = result.get("Internal/External", "Unknown")
            df.at[idx, 'inex_justification'] = result.get("justification", "")
            df.at[idx, 'inex_note'] = result.get("note", "")
            df.at[idx, 'inex_confidence'] = result.get("confidence", 0)

            # Small delay to avoid rate limiting
            time.sleep(AnnotationConfig.REQUEST_DELAY)

        # Save the output file
        print(f"Saving results to {output_file_path}...")
        df.to_csv(output_file_path, index=False)
        print(f"Processing complete! Results saved to {output_file_path}")

        # Print summary
        classification_counts = df['internal vs external'].value_counts()
        print("\nClassification Summary:")
        for classification, count in classification_counts.items():
            print(f"  {classification}: {count}")


def main():
    """Main function to handle command line arguments and run the annotation."""
    parser = argparse.ArgumentParser(description='Annotate data with Internal/External classification using Claude')
    parser.add_argument('input_file', help='Input CSV file path (relative to Claude annotations directory)')
    parser.add_argument('--output_dir', default=None,
                        help=f'Output directory (default: {DIR_ANNOTATIONS_CLAUDE})')

    args = parser.parse_args()

    # Construct full paths using config
    input_path = DIR_ANNOTATIONS_CLAUDE / args.input_file

    # Generate output filename
    input_stem = Path(args.input_file).stem
    output_filename = f"inex_{input_stem}.csv"

    # Use config directory if no output directory specified
    output_dir = Path(args.output_dir) if args.output_dir else DIR_ANNOTATIONS_CLAUDE
    output_path = output_dir / output_filename

    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize annotator and process file
        annotator = InternalExternalAnnotator()
        annotator.process_file(input_path, output_path)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()