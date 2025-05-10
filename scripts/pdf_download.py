#!/usr/bin/env python3
"""
download_fda_pdfs.py  – fetch every Product-Labeling PDF referenced in the
cleaned spreadsheet that already contains a canon_id like “NDA_050441_0086”.
"""

import re, requests, shutil
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
SRC_XLSX = Path("/Users/srinivasana/Documents/peds_agents/data/web_fdaaa_clean.xlsx")
PDF_DIR  = Path("/Users/srinivasana/Documents/peds_agents/pdf")
PDF_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(SRC_XLSX, sheet_name=0, engine="openpyxl")
df.columns = df.columns.str.replace(r"\ufeff", "", regex=True).str.strip()

LINK_COL, ID_COL = "Product Labeling Link", "canon_id"

# ───────── HTTP session
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36")
session = requests.Session()
session.headers.update({"User-Agent": UA})

def download_pdf(url: str, out_path: Path) -> bool:
    try:
        with session.get(url, timeout=(6, 30), stream=True) as r:
            if r.ok and r.headers.get("content-type", "").startswith("application/pdf"):
                with open(out_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
                return True
    except requests.exceptions.RequestException:
        pass
    return False

# ───────── iterate
total = ok = fail = 0

for _, row in df.iterrows():
    raw_links = str(row.get(LINK_COL, "")).strip()
    if not raw_links or not raw_links.lower().startswith("http"):
        continue

    links = re.split(r'[,\n]+', raw_links)          # handle multi-URL cells
    canon = row[ID_COL]                             # e.g. NDA_050441_0086

    for idx, url in enumerate(filter(None, links), start=1):
        total += 1
        base = canon
        if len(links) > 1:                          # second/third link in cell
            base += f"_{idx}"
        out = PDF_DIR / f"{base}.pdf"

        if out.exists():
            print(f"✓ already have {out.name}")
            ok += 1
            continue

        print(f"→ downloading {out.name} … ", end="", flush=True)
        if download_pdf(url.strip(), out):
            print("done")
            ok += 1
        else:
            print("FAILED")
            fail += 1

print(f"\nCompleted. {ok}/{total} PDFs retrieved; {fail} failed.")

