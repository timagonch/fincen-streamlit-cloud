# pipeline.py
from pathlib import Path
import shutil
import pandas as pd

from src.scrape_fincen import scrape
from src.extract_text import batch_extract
from src.fraud_detector import detect

OUT_CSV = "outputs/fraud_articles.csv"

def main():
    print("\n0) clean previous outputs")
    for d in ["data/extracted_text", "outputs"]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    print("1) scrape")
    downloaded = scrape(out_dir="data/raw_pdfs")
    print(f"   downloaded: {downloaded}")

    print("2) extract text")
    extracted = batch_extract(in_dir="data/raw_pdfs", out_dir="data/extracted_text", max_chars=20000)
    print(f"   extracted: {len(extracted)}")

    print("3) detect (TF-IDF keywords + flag sentences)")
    df = detect(txt_dir="data/extracted_text", top_k=8, min_df=2)

    print("4) write dataset")
    Path("outputs").mkdir(exist_ok=True, parents=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"done -> {OUT_CSV}")

if __name__ == "__main__":
    main()
