import pandas as pd
import ast
import os

# Load the analyzed file
df = pd.read_excel('output_metadiscourse_analysis_3.xlsx')

# Parse the stringified list of dictionaries
df['expressions'] = df['Metadiscourse Analysis'].apply(lambda x: ast.literal_eval(f"[{x}]") if pd.notnull(x) else [])

# Expand the expressions list into separate rows
df_exploded = df.explode('expressions')

# Normalize the dictionary values into separate columns
expression_df = pd.json_normalize(df_exploded['expressions'])

# Drop the old expressions column and combine with the new normalized columns
df_exploded = df_exploded.drop(columns=['expressions']).reset_index(drop=True)
final_df = pd.concat([df_exploded, expression_df], axis=1)

# Ensure 'result' folder exists
output_folder = "result"
os.makedirs(output_folder, exist_ok=True)

# Save the processed DataFrame to the result folder
output_path = os.path.join(output_folder, "output_metadiscourse_analysis_expanded.xlsx")
final_df.to_excel(output_path, index=False)

print(f"✅ Expanded analysis saved to '{output_path}'")
