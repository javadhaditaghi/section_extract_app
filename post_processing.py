import pandas as pd
import ast
import os

# Load the analyzed file
df = pd.read_excel('result/output_metadiscourse_analysis_3.xlsx')

# Parse the stringified dict or list of dicts correctly
def safe_parse(x):
    try:
        return ast.literal_eval(x) if pd.notnull(x) else []
    except:
        return []

df['expressions'] = df['Metadiscourse Analysis'].apply(safe_parse)

# If any item is a dict, convert to a list of one item
df['expressions'] = df['expressions'].apply(lambda x: [x] if isinstance(x, dict) else x)

# Expand expressions into separate rows
df_exploded = df.explode('expressions')

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
