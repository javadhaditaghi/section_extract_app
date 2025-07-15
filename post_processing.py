import pandas as pd
import ast


df = pd.read_excel('output_metadiscourse_analysis_3.xlsx')


df['expressions'] = df['Metadiscourse Analysis'].apply(lambda x: ast.literal_eval(f"[{x}]") if pd.notnull(x) else [])


df_exploded = df.explode('expressions')

expression_df = pd.json_normalize(df_exploded['expressions'])

# Drop the old column and join new expanded columns
df_exploded = df_exploded.drop(columns=['expressions']).reset_index(drop=True)
final_df = pd.concat([df_exploded, expression_df], axis=1)

# Save or display
final_df.to_csv('formatted_output.csv', index=False)
