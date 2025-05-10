#!/usr/bin/env python3
"""make_txt_subset.py – copy the TXT files for the manually annotated set."""

import shutil, pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
TXT_SRC  = Path("/Users/srinivasana/Documents/peds_agents_local/txt")   # all txt
TXT_DST  = Path("/Users/srinivasana/Documents/peds_agents_local/txt_test_subset")
TXT_DST.mkdir(parents=True, exist_ok=True)

# read the manual-annotation sheet (already cleaned, with canon_id column)
manual = pd.read_excel(
    "/Users/srinivasana/Documents/peds_agents_local/data/manual_annotated_labels.xlsx",
    sheet_name=0,
    dtype=str, keep_default_na=False
)

canon_set = set(manual["canon_id"].dropna())

# ── copy matching files ────────────────────────────────────────────────
copied = skipped = 0
for txt_path in TXT_SRC.glob("*.txt"):
    if txt_path.stem in canon_set:
        shutil.copy2(txt_path, TXT_DST / txt_path.name)
        copied += 1
    else:
        skipped += 1

print(f"Copied {copied} files ➜ {TXT_DST}")
print(f"Skipped {skipped} other TXT files")
