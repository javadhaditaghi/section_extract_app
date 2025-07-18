# import pandas as pd
# import ast
# import os
#
# # Load the analyzed file
# df = pd.read_excel('result/output_metadiscourse_analysis_3.xlsx')
#
# # Parse the stringified dict or list of dicts correctly
# def safe_parse(x):
#     try:
#         return ast.literal_eval(x) if pd.notnull(x) else []
#     except:
#         return []
#
# df['expressions'] = df['Metadiscourse Analysis'].apply(safe_parse)
#
# # If any item is a dict, convert to a list of one item
# df['expressions'] = df['expressions'].apply(lambda x: [x] if isinstance(x, dict) else x)
#
# # Expand expressions into separate rows
# df_exploded = df.explode('expressions')
#
# # Normalize the dictionary values into separate columns
# expression_df = pd.json_normalize(df_exploded['expressions'])
#
# # Drop the old expressions column and combine with the normalized one
# df_exploded = df_exploded.drop(columns=['expressions']).reset_index(drop=True)
# final_df = pd.concat([df_exploded, expression_df], axis=1)
#
# # Ensure 'result' folder exists
# output_folder = "result"
# os.makedirs(output_folder, exist_ok=True)
#
# # Save the processed DataFrame
# output_path = os.path.join(output_folder, "output_metadiscourse_analysis_expanded.xlsx")
# final_df.to_excel(output_path, index=False)
#
# print(f"✅ Expanded analysis saved to '{output_path}'")


# import pandas as pd
# import ast
# import json
# import os
# import re
#
# # Load the analyzed file
# df = pd.read_excel('result/output_metadiscourse_analysis_3.xlsx')
#
#
# # Improved parsing function that handles multiple formats
# def safe_parse(x):
#     if pd.isnull(x):
#         return []
#
#     try:
#         # First, try to parse as-is (in case it's already a proper JSON array)
#         return ast.literal_eval(x)
#     except:
#         try:
#             # Try to parse with json.loads
#             return json.loads(x)
#         except:
#             try:
#                 # If it's multiple dict objects separated by commas, wrap in brackets
#                 x_str = str(x).strip()
#                 if x_str.startswith('{') and x_str.endswith('}'):
#                     # Check if it contains multiple objects by counting opening braces
#                     if x_str.count('}, {') > 0:
#                         # Wrap in square brackets to make it a proper JSON array
#                         x_str = '[' + x_str + ']'
#                     else:
#                         # Single object, wrap in brackets
#                         x_str = '[' + x_str + ']'
#
#                 return json.loads(x_str)
#             except:
#                 try:
#                     # Try ast.literal_eval on the bracketed version
#                     return ast.literal_eval(x_str)
#                 except:
#                     print(f"Warning: Could not parse: {x}")
#                     return []
#
#
# # Apply the parsing function
# df['expressions'] = df['Metadiscourse Analysis'].apply(safe_parse)
#
# # Ensure all items are lists (convert single dict to list)
# df['expressions'] = df['expressions'].apply(lambda x: [x] if isinstance(x, dict) else x)
#
# # Filter out empty lists to avoid issues
# df = df[df['expressions'].apply(len) > 0]
#
# # Expand expressions into separate rows
# df_exploded = df.explode('expressions')
#
# # Remove rows where expressions is None or empty
# df_exploded = df_exploded[df_exploded['expressions'].notna()]
#
# # Normalize the dictionary values into separate columns
# expression_df = pd.json_normalize(df_exploded['expressions'])
#
# # Drop the old expressions column and combine with the normalized one
# df_exploded = df_exploded.drop(columns=['expressions']).reset_index(drop=True)
# final_df = pd.concat([df_exploded, expression_df], axis=1)
#
# # Ensure 'result' folder exists
# output_folder = "result"
# os.makedirs(output_folder, exist_ok=True)
#
# # Save the processed DataFrame
# output_path = os.path.join(output_folder, "output_metadiscourse_analysis_expanded.xlsx")
# final_df.to_excel(output_path, index=False)
#
# print(f"✅ Expanded analysis saved to '{output_path}'")
# print(f"📊 Total rows in final dataset: {len(final_df)}")
# print(f"📊 Columns: {list(final_df.columns)}")
#
# # Show a preview of the results
# if len(final_df) > 0:
#     print("\n📋 Preview of expanded data:")
#     print(final_df[['expression', 'confidence', 'justification']].head())


import pandas as pd
import ast
import json
import os
import re
import glob


def get_input_file():
    """Allow user to choose input file interactively"""
    # Look for Excel files in the result folder
    result_files = glob.glob('result/*.xlsx')

    if not result_files:
        print("❌ No Excel files found in the 'result' folder.")
        return None

    print("📁 Available Excel files in 'result' folder:")
    for i, file in enumerate(result_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    print(f"{len(result_files) + 1}. Enter custom file path")

    while True:
        try:
            choice = input(f"\nChoose a file (1-{len(result_files) + 1}): ").strip()

            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(result_files):
                    return result_files[choice_num - 1]
                elif choice_num == len(result_files) + 1:
                    custom_path = input("Enter the full path to your Excel file: ").strip()
                    if os.path.exists(custom_path) and custom_path.endswith('.xlsx'):
                        return custom_path
                    else:
                        print("❌ File not found or not an Excel file. Please try again.")
                else:
                    print(f"❌ Please enter a number between 1 and {len(result_files) + 1}")
            else:
                print("❌ Please enter a valid number")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return None


def safe_parse(x):
    """Improved parsing function that handles multiple formats"""
    if pd.isnull(x):
        return []

    try:
        # First, try to parse as-is (in case it's already a proper JSON array)
        return ast.literal_eval(x)
    except:
        try:
            # Try to parse with json.loads
            return json.loads(x)
        except:
            try:
                # If it's multiple dict objects separated by commas, wrap in brackets
                x_str = str(x).strip()
                if x_str.startswith('{') and x_str.endswith('}'):
                    # Check if it contains multiple objects by counting opening braces
                    if x_str.count('}, {') > 0:
                        # Wrap in square brackets to make it a proper JSON array
                        x_str = '[' + x_str + ']'
                    else:
                        # Single object, wrap in brackets
                        x_str = '[' + x_str + ']'

                return json.loads(x_str)
            except:
                try:
                    # Try ast.literal_eval on the bracketed version
                    return ast.literal_eval(x_str)
                except:
                    print(f"Warning: Could not parse: {x}")
                    return []


def process_metadiscourse_file(input_file):
    """Process the selected metadiscourse analysis file"""
    try:
        # Load the analyzed file
        print(f"📖 Loading file: {input_file}")
        df = pd.read_excel(input_file)

        # Apply the parsing function
        df['expressions'] = df['Metadiscourse Analysis'].apply(safe_parse)

        # Ensure all items are lists (convert single dict to list)
        df['expressions'] = df['expressions'].apply(lambda x: [x] if isinstance(x, dict) else x)

        # Filter out empty lists to avoid issues
        df = df[df['expressions'].apply(len) > 0]

        # Expand expressions into separate rows
        df_exploded = df.explode('expressions')

        # Remove rows where expressions is None or empty
        df_exploded = df_exploded[df_exploded['expressions'].notna()]

        # Normalize the dictionary values into separate columns
        expression_df = pd.json_normalize(df_exploded['expressions'])

        # Drop the old expressions column and combine with the normalized one
        df_exploded = df_exploded.drop(columns=['expressions']).reset_index(drop=True)
        final_df = pd.concat([df_exploded, expression_df], axis=1)

        # Generate output filename based on input filename
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = f"{input_basename}_expanded.xlsx"

        # Ensure 'result' folder exists
        output_folder = "result"
        os.makedirs(output_folder, exist_ok=True)

        # Save the processed DataFrame
        output_path = os.path.join(output_folder, output_filename)
        final_df.to_excel(output_path, index=False)

        print(f"✅ Expanded analysis saved to '{output_path}'")
        print(f"📊 Total rows in final dataset: {len(final_df)}")
        print(f"📊 Columns: {list(final_df.columns)}")

        # Show a preview of the results
        if len(final_df) > 0:
            print("\n📋 Preview of expanded data:")
            print(final_df[['expression', 'confidence', 'justification']].head())

        return output_path

    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return None


def main():
    """Main function to run the interactive processor"""
    print("🔍 Metadiscourse Analysis Processor")
    print("=" * 40)

    # Get input file from user
    input_file = get_input_file()

    if input_file:
        # Process the selected file
        output_path = process_metadiscourse_file(input_file)

        if output_path:
            print(f"\n🎉 Processing completed successfully!")
            print(f"📄 Output saved to: {output_path}")
        else:
            print("\n❌ Processing failed. Please check the file format and try again.")
    else:
        print("\n❌ No file selected. Exiting.")


if __name__ == "__main__":
    main()