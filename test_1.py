import pdfplumber
from transformers import pipeline

thesis_name = "tesi_Lazzeretti_finale.pdf"

name_part = thesis_name.split('.')[0]
new_name = name_part + ".txt"


with pdfplumber.open(thesis_name) as pdf, open(new_name, "w", encoding="utf-8") as f:
  for page in pdf.pages:
    t = page.extract_text()
    if t:
      f.write(t + '\n')
