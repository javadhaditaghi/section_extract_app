import pandas as pd
import google.generativeai as genai
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-pro")

# Load Excel files
openai_df = pd.read_excel("result/output_metadiscourse_analysis_3_expanded.xlsx")
gemini_df = pd.read_excel("result/output_metadiscourse_analysis_gemini_expanded.xlsx")

# Standardize necessary columns
key_columns = ["index", "sentence", "section", "thesis code"]
openai_df["source"] = "OpenAI"
gemini_df["source"] = "Gemini"
all_data = pd.concat([openai_df, gemini_df], ignore_index=True)

# Group by sentence (and metadata)
grouped = all_data.groupby(key_columns)

# Result storage
results = []

# Compare all annotations per sentence
for key, group in grouped:
    sentence = key[1]
    openai_rows = group[group["source"] == "OpenAI"]
    gemini_rows = group[group["source"] == "Gemini"]

    for _, openai_row in openai_rows.iterrows():
        prompt = f"""
You are an expert in academic discourse. Compare the following annotations for the sentence and decide which one is the best in terms of rhetorical appropriateness and justification. Choose from OpenAI or Gemini. Provide your decision and justification.

Sentence:
"{sentence}"

Annotation from OpenAI:
Expression: "{openai_row['expression']}"
Confidence: {openai_row['confidence']}
Justification: "{openai_row['justification']}"

"""
        for i, (_, gem_row) in enumerate(gemini_rows.iterrows(), 1):
            prompt += f"""
Annotation {i} from Gemini:
Expression: "{gem_row['expression']}"
Confidence: {gem_row['confidence']}
Justification: "{gem_row['justification']}"
"""

        prompt += "\nWhich annotation is best overall (OpenAI or Gemini), and why?"

        try:
            time.sleep(1.5)
            response = model.generate_content(prompt)
            decision = response.text.strip()
            if "openai" in decision.lower():
                chosen_source = "OpenAI"
                chosen_expression = openai_row["expression"]
                chosen_justification = openai_row["justification"]
            elif "gemini" in decision.lower():
                chosen_source = "Gemini"
                # Optional: extract which Gemini annotation was chosen
                chosen_expression = "Unclear"
                chosen_justification = "Unclear"
            else:
                chosen_source = "Unclear"
                chosen_expression = ""
                chosen_justification = ""

            result_row = {
                **{col: openai_row[col] for col in key_columns},
                "sentence": sentence,
                "openai_expression": openai_row["expression"],
                "openai_justification": openai_row["justification"],
                "openai_confidence": openai_row["confidence"],
                "gemini_expressions": "; ".join(gemini_rows["expression"]),
                "gemini_justifications": "; ".join(gemini_rows["justification"]),
                "gemini_confidences": "; ".join(map(str, gemini_rows["confidence"])),
                "chosen_source": chosen_source,
                "chosen_expression": chosen_expression,
                "chosen_justification": chosen_justification,
                "gemini_rationale": decision
            }

            results.append(result_row)

        except Exception as e:
            results.append({
                **{col: openai_row[col] for col in key_columns},
                "sentence": sentence,
                "openai_expression": openai_row["expression"],
                "openai_justification": openai_row["justification"],
                "openai_confidence": openai_row["confidence"],
                "gemini_expressions": "; ".join(gemini_rows["expression"]),
                "gemini_justifications": "; ".join(gemini_rows["justification"]),
                "gemini_confidences": "; ".join(map(str, gemini_rows["confidence"])),
                "chosen_source": "Error",
                "chosen_expression": "",
                "chosen_justification": "",
                "gemini_rationale": str(e)
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to Excel
output_path = "result/optimized_metadiscourse_annotations_by_sentence.xlsx"
results_df.to_excel(output_path, index=False)

output_path
