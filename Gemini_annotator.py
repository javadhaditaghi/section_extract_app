

# Annotator.py - Updated for Google Gemini API

import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.5-pro')

# === Load your Excel or CSV file ===
# Example: 'input.xlsx' with a column named 'Sentence'
file_path = "datasheets//ALL_DATASHEETS.csv"  # or "input.csv"
df = pd.read_csv(file_path)  # use pd.read_csv() if using CSV

# Limit to first 10 rows
df = df.head(2)

# Add a new column to store the output
df["Metadiscourse Analysis"] = ""

# === Step 1 Prompt ===
system_prompt = """You are an academic language expert assisting with the identification of potential metadiscourse expressions in academic writing, based on Hyland's (2005) model.
Your task is to read the sentence carefully in its surrounding context and mark any expression that might serve a metadiscursive function, even if you're unsure about the exact category or if it turns out to be propositional. Err on the side of inclusion.
Focus on lexical indicators and rhetorical intent, not surface form or part of speech alone. Use the function of the expression in context to decide whether it may contribute to the reader–writer relationship (interactional) or text organization (interactive).

Instructions:
Mark any expression in the sentence that could reasonably function as metadiscourse, based on its context and rhetorical purpose — not just the word list.

Do not classify the expression yet (e.g., hedge, booster, etc.). That comes later.

Skip routine, factual, or propositional uses of common words unless they serve a rhetorical function.

Use the surrounding context to judge whether the expression:

Organizes the discourse (interactive), or

Positions the writer and reader (interactional).

Avoid shallow keyword spotting. Judge by rhetorical intent, not just presence of a word.
Use the keyword reference list below to help you detect common indicators — but never rely on keyword spotting alone.


Output for each sentence should include:


Each expression identified

A confidence score (1–5)

Optional note if unsure (e.g., "I'm not sure" or "borderline case")

A brief justification based on its rhetorical role

Use this list to guide your attention — these are common, not exhaustive:


Interactional 
Hedges
almost, apparently, appear to be, approximately, assume, believed, certain extent, certain level, certain amount, could, couldn't, doubt, essentially, estimate, frequently, generally, in general, indicate, largely, likely, mainly, may, maybe, might, mostly, often, perhaps, plausible, possible, possibly, presumably, probable, probably, relatively, seems, sometimes, somewhat, suggest, suspect, unlikely, uncertain, unclear, usually, would, wouldn't, little, not understood

Boosters (Emphatics)
actually, always, apparent, certain that, certainly, certainty, clearly, it is clear, conclusively, decidedly, definitely, demonstrate, determine, doubtless, essential, establish, in fact, the fact that, indeed, know, it is known that, must, never, no doubt, beyond doubt, obvious, obviously, of course, prove, show, sure, true, undoubtedly, well known, won't, even if, should, by far

Attitude Markers
admittedly, I agree, amazingly, appropriately, correctly, curiously, disappointing, disagree, even, fortunately, have to, hopefully, important, importantly, interest, interestingly, prefer, pleased, must, ought, remarkable, surprisingly, unfortunate, unfortunately, unusually, understandably

Engagement Markers (Relational Markers)
incidentally, by the way, consider, imagine, let, let us, let's, lets, our, recall, us, we, you, next, to begin, note, notice, your, one's, think about

Self-Mention (Person Markers)
I, we, me, my, our, mine


Interactive 
Transitions (Logical Connectives)
and, but, therefore, thereby, so as to, in addition, similarly, equally, likewise, moreover, furthermore, in contrast, by contrast, as a result, the result is, result in, since, because, consequently, as a consequence, accordingly, on the other hand, on the contrary, however, also, yet, or

Frame Markers (Announce Goals)
my purpose, the aim, I intend, I seek, I wish, I argue, I propose, I suggest, I discuss, I would like to, I will focus on, we will focus on, I will emphasise, we will emphasise, my goal is, in this section, in this chapter, here I do this, here I will

Frame Markers (Label Stages)
to conclude, in conclusion, to sum up, in sum, summarise, summarize, overall, on the whole, all in all, so far, thus far, to repeat

Frame Markers (Sequencing)
to start with, first, firstly, second, secondly, third, thirdly, fourth, fourthly, fifthly, next, last, two, three, four, five

Output Format (for each sentence):
Provide the output as a list of identified expressions in the following structure for each entry (exactly this structure; do not add anything extra):
{
    "expression": "<highlighted metadiscourse expression>",
    "confidence": <1–5>,
    "note": "<optional note if unsure, else leave blank>",
    "justification": "<brief reason based on rhetorical intent>"
  },
  {
    "expression_2": "<2nd highlighted metadiscourse expression>",
    "confidence": <1–5>,
    "note": "<optional note if unsure, else leave blank>",
    "justification": "<brief reason based on rhetorical intent>"
  },

"""


# === Function to call Gemini API ===
def analyze_metadiscourse(sentence):
    try:
        # Create the full prompt with system instructions and user input
        full_prompt = f"{system_prompt}\n\nSentence: {sentence}"

        # Generate response using Gemini
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=2048,
            )
        )

        return response.text.strip()

    except Exception as e:
        return f"ERROR: {e}"


# === Iterate over sentences and apply the analysis ===
for idx in tqdm(df.index, desc="Analyzing Sentences"):
    sentence = df.at[idx, "sentence"]
    result = analyze_metadiscourse(sentence)
    df.at[idx, "Metadiscourse Analysis"] = result

# output inside a folder called result
output_folder = "result"
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, "output_metadiscourse_analysis_gemini.xlsx")
df.to_excel(output_path, index=False)

print("✅ Analysis completed and saved to 'output_metadiscourse_analysis_gemini.xlsx'")