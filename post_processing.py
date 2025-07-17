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


import pandas as pd
import ast
import json
import os
import re

# Load the analyzed file
df = pd.read_excel('result/output_metadiscourse_analysis_3.xlsx')


# Improved parsing function that handles multiple formats
def safe_parse(x):
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

# Ensure 'result' folder exists
output_folder = "result"
os.makedirs(output_folder, exist_ok=True)

# Save the processed DataFrame
output_path = os.path.join(output_folder, "output_metadiscourse_analysis_expanded.xlsx")
final_df.to_excel(output_path, index=False)

print(f"✅ Expanded analysis saved to '{output_path}'")
print(f"📊 Total rows in final dataset: {len(final_df)}")
print(f"📊 Columns: {list(final_df.columns)}")

# Show a preview of the results
if len(final_df) > 0:
    print("\n📋 Preview of expanded data:")
    print(final_df[['expression', 'confidence', 'justification']].head())