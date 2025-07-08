# run_merge.py
from pathlib import Path
import pandas as pd
from config import DIR_DATASHEETS

def merge_datasheets() -> Path:
    merged = pd.DataFrame()
    files = list(DIR_DATASHEETS.glob("*_datasheet.csv"))

    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name   # optional: track original file
        merged = pd.concat([merged, df], ignore_index=True)

    out_path = DIR_DATASHEETS / "ALL_DATASHEETS.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Merged datasheet saved to {out_path}")
    return out_path

if __name__ == "__main__":
    merge_datasheets()
