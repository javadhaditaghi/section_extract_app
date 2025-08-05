# src/postprocessing/expand_annotation.py
"""
Enhanced Metadiscourse Analysis Processor

This module processes metadiscourse analysis results from various LLM providers
(GPT, Claude, DeepSeek, Gemini) and expands structured annotations into
separate rows for detailed analysis.

Usage:
    python src/postprocessing/expand_annotation.py
"""

import pandas as pd
import ast
import json
import os
import re
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any
import traceback

# Import project configuration
from src.utils.config import DIR_ANNOTATIONS_LLM


def get_available_files() -> List[str]:
    """Get all CSV files from LLM annotation directories"""
    annotation_dirs = [
        DIR_ANNOTATIONS_LLM / "GPT",
        DIR_ANNOTATIONS_LLM / "Claude",
        DIR_ANNOTATIONS_LLM / "DeepSeek",
        DIR_ANNOTATIONS_LLM / "Gemini"
    ]

    csv_files = []
    for directory in annotation_dirs:
        if directory.exists():
            pattern = str(directory / "*.csv")
            csv_files.extend(glob.glob(pattern))

    return csv_files


def get_input_file() -> Optional[str]:
    """Allow user to choose input file from LLM annotation directories"""
    csv_files = get_available_files()

    if not csv_files:
        print("❌ No CSV files found in LLM annotation directories.")
        print("💡 Make sure you have run annotation scripts first:")
        print("   - python run_gpt_annotation.py")
        print("   - python run_claude_annotation.py")
        print("   - python run_deepseek_annotation.py")
        return None

    print("📁 Available CSV files from LLM annotations:")
    print("=" * 60)

    # Group files by provider for better display
    providers = {}
    for file_path in csv_files:
        path_obj = Path(file_path)
        provider = path_obj.parent.name
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(file_path)

    file_list = []
    for provider, files in providers.items():
        print(f"\n🤖 {provider}:")
        for file_path in files:
            file_list.append(file_path)
            filename = Path(file_path).name
            print(f"   {len(file_list)}. {filename}")

    print("=" * 60)

    while True:
        try:
            choice = input(f"\nChoose a file (1-{len(file_list)}): ").strip()

            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(file_list):
                    return file_list[choice_num - 1]
                else:
                    print(f"❌ Please enter a number between 1 and {len(file_list)}")
            else:
                print("❌ Please enter a valid number")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Operation cancelled.")
            return None


def clean_markdown_json(text: str) -> str:
    """Remove markdown code block formatting from JSON text"""
    if pd.isnull(text):
        return ""

    text_str = str(text).strip()

    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    if text_str.startswith('```'):
        lines = text_str.split('\n')
        if len(lines) > 1:
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's just ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text_str = '\n'.join(lines)

    return text_str.strip()


def safe_parse(x) -> List[Dict[str, Any]]:
    """Enhanced parsing function that handles multiple formats including markdown-wrapped JSON"""
    if pd.isnull(x) or x == "":
        return []

    # First, clean any markdown formatting
    x_cleaned = clean_markdown_json(x)

    if not x_cleaned:
        return []

    try:
        # Try to parse as JSON first
        if x_cleaned.startswith('[') and x_cleaned.endswith(']'):
            return json.loads(x_cleaned)
        elif x_cleaned.startswith('{') and x_cleaned.endswith('}'):
            # Single object, wrap in array
            return [json.loads(x_cleaned)]
    except json.JSONDecodeError:
        pass

    try:
        # Try ast.literal_eval
        result = ast.literal_eval(x_cleaned)
        if isinstance(result, dict):
            return [result]
        elif isinstance(result, list):
            return result
        else:
            return []
    except (ValueError, SyntaxError):
        pass

    try:
        # Try to fix common JSON issues
        x_fixed = x_cleaned.replace("'", '"')  # Replace single quotes
        x_fixed = re.sub(r'(\w+):', r'"\1":', x_fixed)  # Quote unquoted keys

        if x_fixed.startswith('{') and x_fixed.endswith('}'):
            return [json.loads(x_fixed)]
        elif x_fixed.startswith('[') and x_fixed.endswith(']'):
            return json.loads(x_fixed)
    except json.JSONDecodeError:
        pass

    # If all parsing fails, try to extract any structured information
    try:
        # Updated patterns to match the new structure
        expressions = []
        expression_pattern = r'expression["\s]*:[\s]*["\']([^"\']+)["\']'
        confidence_pattern = r'(?<!inex_)confidence["\s]*:[\s]*([0-9.]+)'
        justification_pattern = r'(?<!inex_)justification["\s]*:[\s]*["\']([^"\']*)["\']'
        note_pattern = r'(?<!inex_)note["\s]*:[\s]*["\']([^"\']*)["\']'
        internal_external_pattern = r'internal_external["\s]*:[\s]*["\']([^"\']*)["\']'
        inex_confidence_pattern = r'inex_confidence["\s]*:[\s]*([0-9.]+)'
        inex_note_pattern = r'inex_note["\s]*:[\s]*["\']([^"\']*)["\']'
        inex_justification_pattern = r'inex_justification["\s]*:[\s]*["\']([^"\']*)["\']'

        expressions_found = re.findall(expression_pattern, x_cleaned, re.IGNORECASE)
        confidences_found = re.findall(confidence_pattern, x_cleaned, re.IGNORECASE)
        justifications_found = re.findall(justification_pattern, x_cleaned, re.IGNORECASE)
        notes_found = re.findall(note_pattern, x_cleaned, re.IGNORECASE)
        internal_external_found = re.findall(internal_external_pattern, x_cleaned, re.IGNORECASE)
        inex_confidences_found = re.findall(inex_confidence_pattern, x_cleaned, re.IGNORECASE)
        inex_notes_found = re.findall(inex_note_pattern, x_cleaned, re.IGNORECASE)
        inex_justifications_found = re.findall(inex_justification_pattern, x_cleaned, re.IGNORECASE)

        if expressions_found:
            for i, expr in enumerate(expressions_found):
                conf = float(confidences_found[i]) if i < len(confidences_found) else 0.0
                just = justifications_found[i] if i < len(justifications_found) else ""
                note = notes_found[i] if i < len(notes_found) else ""
                int_ext = internal_external_found[i] if i < len(internal_external_found) else ""
                inex_conf = float(inex_confidences_found[i]) if i < len(inex_confidences_found) else 0.0
                inex_note = inex_notes_found[i] if i < len(inex_notes_found) else ""
                inex_just = inex_justifications_found[i] if i < len(inex_justifications_found) else ""

                expressions.append({
                    'expression': expr,
                    'confidence': conf,
                    'note': note,
                    'justification': just,
                    'internal_external': int_ext,
                    'inex_confidence': inex_conf,
                    'inex_note': inex_note,
                    'inex_justification': inex_just
                })
            return expressions
    except:
        pass

    print(f"⚠️  Warning: Could not parse: {str(x_cleaned)[:100]}...")
    return []


def process_metadiscourse_file(input_file: str) -> Optional[str]:
    """Process the selected metadiscourse analysis file"""
    try:
        input_path = Path(input_file)
        print(f"📖 Loading file: {input_path.name}")
        print(f"🤖 Provider: {input_path.parent.name}")

        # Load the CSV file
        df = pd.read_csv(input_file)

        print(f"📊 Original dataset shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")

        # Check if the required column exists
        if 'Metadiscourse Analysis' not in df.columns:
            print("❌ 'Metadiscourse Analysis' column not found in the file.")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Check for error entries first
        error_count = df['Metadiscourse Analysis'].astype(str).str.contains('ERROR', case=False, na=False).sum()
        if error_count > 0:
            print(f"⚠️  Found {error_count} error entries in the analysis")

        # Apply the parsing function
        print("🔄 Parsing metadiscourse analysis data...")
        df['expressions'] = df['Metadiscourse Analysis'].apply(safe_parse)

        # Show parsing statistics
        total_rows = len(df)
        parsed_rows = len(df[df['expressions'].apply(len) > 0])
        print(f"📊 Successfully parsed {parsed_rows} out of {total_rows} rows ({parsed_rows / total_rows * 100:.1f}%)")

        # Ensure all items are lists
        df['expressions'] = df['expressions'].apply(
            lambda x: [x] if isinstance(x, dict) else (x if isinstance(x, list) else [])
        )

        # Filter out empty lists
        df_filtered = df[df['expressions'].apply(len) > 0].copy()
        print(f"📊 Rows with valid expressions: {len(df_filtered)}")

        if len(df_filtered) == 0:
            print("❌ No valid expressions found after parsing.")
            print("💡 This might happen if:")
            print("   - The analysis contains mostly errors")
            print("   - The JSON format is not as expected")
            print("   - The model output format has changed")
            return None

        # Expand expressions into separate rows
        print("🔄 Expanding expressions into separate rows...")
        df_exploded = df_filtered.explode('expressions').reset_index(drop=True)

        # Remove rows where expressions is None or empty
        df_exploded = df_exploded[df_exploded['expressions'].notna()]

        # Check if we still have data
        if len(df_exploded) == 0:
            print("❌ No data remaining after explosion.")
            return None

        # Normalize the dictionary values into separate columns
        print("🔄 Normalizing expression data...")
        try:
            expression_df = pd.json_normalize(df_exploded['expressions'])

            # Updated expected columns to match the new structure
            expected_columns = [
                'expression', 'confidence', 'note', 'justification',
                'internal_external', 'inex_confidence', 'inex_note', 'inex_justification'
            ]
            for col in expected_columns:
                if col not in expression_df.columns:
                    # Set appropriate default values based on column type
                    if 'confidence' in col:
                        expression_df[col] = 0.0
                    else:
                        expression_df[col] = ""
                    print(f"📝 Added missing column: {col}")

            # Remove any unwanted columns like 'source'
            unwanted_columns = ['source']
            for col in unwanted_columns:
                if col in expression_df.columns:
                    expression_df = expression_df.drop(columns=[col])
                    print(f"🗑️  Removed unwanted column: {col}")

        except Exception as e:
            print(f"❌ Error normalizing expression data: {e}")
            print("💡 Trying alternative approach...")

            # Alternative: manually extract common fields
            expression_data = []
            for expr in df_exploded['expressions']:
                if isinstance(expr, dict):
                    # Extract specific fields we want with the new structure
                    cleaned_expr = {
                        'expression': expr.get('expression', ''),
                        'confidence': expr.get('confidence', 0.0),
                        'note': expr.get('note', ''),
                        'justification': expr.get('justification', ''),
                        'internal_external': expr.get('internal_external', ''),
                        'inex_confidence': expr.get('inex_confidence', 0.0),
                        'inex_note': expr.get('inex_note', ''),
                        'inex_justification': expr.get('inex_justification', '')
                    }
                    expression_data.append(cleaned_expr)
                else:
                    expression_data.append({
                        'expression': str(expr),
                        'confidence': 0.0,
                        'note': '',
                        'justification': '',
                        'internal_external': '',
                        'inex_confidence': 0.0,
                        'inex_note': '',
                        'inex_justification': ''
                    })

            expression_df = pd.DataFrame(expression_data)

        # Combine with original data (drop expressions and Metadiscourse Analysis columns)
        df_exploded = df_exploded.drop(columns=['expressions'])

        # Also remove the original Metadiscourse Analysis column since we've extracted its data
        if 'Metadiscourse Analysis' in df_exploded.columns:
            df_exploded = df_exploded.drop(columns=['Metadiscourse Analysis'])
            print("🗑️  Removed original 'Metadiscourse Analysis' column")

        final_df = pd.concat([df_exploded.reset_index(drop=True),
                              expression_df.reset_index(drop=True)], axis=1)

        # Generate output filenames in the same directory
        output_filename = f"cleaned_{input_path.stem}.csv"
        output_path = input_path.parent / output_filename

        narrow_output_filename = f"narrow_cleaned_{input_path.stem}.csv"
        narrow_output_path = input_path.parent / narrow_output_filename

        # Save the full cleaned CSV
        final_df.to_csv(output_path, index=False)
        print(f"✅ Full expanded analysis saved to: {output_path}")

        # Create narrow version by dropping specified columns
        columns_to_drop = ['source_file', 'source_stem', 'index', 'file_order', 'annotation_timestamp']

        # Only drop columns that actually exist
        existing_columns_to_drop = [col for col in columns_to_drop if col in final_df.columns]
        narrow_df = final_df.drop(columns=existing_columns_to_drop)

        if existing_columns_to_drop:
            print(f"🗑️  Dropped columns for narrow version: {existing_columns_to_drop}")

        # Save the narrow cleaned CSV
        narrow_df.to_csv(narrow_output_path, index=False)
        print(f"✅ Narrow expanded analysis saved to: {narrow_output_path}")

        print(f"📊 Total rows in both datasets: {len(final_df)}")
        print(f"📊 Full dataset columns ({len(final_df.columns)}): {list(final_df.columns)}")
        print(f"📊 Narrow dataset columns ({len(narrow_df.columns)}): {list(narrow_df.columns)}")

        # Show a preview of the results
        if len(final_df) > 0:
            print("\n📋 Preview of expanded data:")

            # Try to show relevant columns - updated for new structure
            preview_cols = []
            # Prioritize the main columns we expect
            main_cols = ['expression', 'confidence', 'justification', 'internal_external', 'inex_confidence']
            for col in main_cols:
                if col in final_df.columns:
                    preview_cols.append(col)

            # Add other interesting columns if space allows
            other_cols = ['category', 'type', 'sentence', 'note']
            for col in other_cols:
                if col in final_df.columns and len(preview_cols) < 7:
                    preview_cols.append(col)

            if not preview_cols:
                # Show first few columns if no standard ones found
                preview_cols = list(final_df.columns)[:7]

            print(final_df[preview_cols].head(3))

            # Show statistics for key columns
            if 'expression' in final_df.columns:
                unique_expressions = final_df['expression'].nunique()
                total_expressions = len(final_df)
                print(f"\n📊 Total expressions: {total_expressions}")
                print(f"📊 Unique expressions: {unique_expressions}")

            if 'confidence' in final_df.columns:
                avg_confidence = final_df['confidence'].mean()
                print(f"📊 Average confidence: {avg_confidence:.2f}")

            # Show statistics for new internal/external fields
            if 'internal_external' in final_df.columns:
                internal_external_counts = final_df['internal_external'].value_counts()
                print(f"📊 Internal/External distribution:")
                for category, count in internal_external_counts.items():
                    if category:  # Only show non-empty categories
                        print(f"   {category}: {count}")

            if 'inex_confidence' in final_df.columns:
                avg_inex_confidence = final_df['inex_confidence'].mean()
                print(f"📊 Average internal/external confidence: {avg_inex_confidence:.2f}")

            if 'justification' in final_df.columns:
                with_justification = final_df['justification'].astype(str).str.len().gt(0).sum()
                print(f"📊 Entries with justification: {with_justification}/{total_expressions}")

            if 'inex_justification' in final_df.columns:
                with_inex_justification = final_df['inex_justification'].astype(str).str.len().gt(0).sum()
                print(f"📊 Entries with internal/external justification: {with_inex_justification}/{total_expressions}")

        return str(output_path), str(narrow_output_path)

    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        print("\n🔍 Full error details:")
        traceback.print_exc()
        return None


def main():
    """Main function to run the interactive processor"""
    print("🔍 Enhanced Metadiscourse Analysis Processor")
    print("=" * 60)
    print("This tool expands LLM annotation results into structured data")
    print("Supports: GPT, Claude, DeepSeek, Gemini annotation outputs")
    print("Updated for new JSON structure with internal/external classification")
    print("=" * 60)

    # Get input file from user
    input_file = get_input_file()

    if input_file:
        print(f"\n🎯 Processing selected file...")
        print(f"📁 Input: {Path(input_file).name}")
        print(f"🤖 Provider: {Path(input_file).parent.name}")

        # Process the selected file
        result = process_metadiscourse_file(input_file)

        if result:
            output_path, narrow_output_path = result
            print(f"\n🎉 Processing completed successfully!")
            print(f"📄 Full output saved to: {output_path}")
            print(f"📄 Narrow output saved to: {narrow_output_path}")
            print(f"\n💡 Next steps:")
            print(f"   • Use full dataset for comprehensive analysis")
            print(f"   • Use narrow dataset for focused analysis")
            print(f"   • Compare results across different LLM providers")
            print(f"   • Analyze internal vs external metadiscourse patterns")
        else:
            print("\n❌ Processing failed.")
            print("💡 Troubleshooting tips:")
            print("   • Check that the input file has 'Metadiscourse Analysis' column")
            print("   • Ensure the analysis results are in proper JSON format")
            print("   • Try running the annotation script again if many errors")
    else:
        print("\n❌ No file selected. Exiting.")


if __name__ == "__main__":
    main()