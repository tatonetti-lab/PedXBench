#!/usr/bin/env python3
"""
pdf_to_txt.py – plain-text extraction with pdfminer.six
"""

from pathlib import Path
import sys
from pdfminer.high_level import extract_text       # << simple API

PDF_DIR = Path("/Users/srinivasana/Documents/peds_agents/pdf")
TXT_DIR = Path("/Users/srinivasana/Documents/peds_agents/txt")
TXT_DIR.mkdir(parents=True, exist_ok=True)

pdf_files = sorted(PDF_DIR.glob("*.pdf"))
print(f"{len(pdf_files)} PDFs found.")

ok = fail = 0
for pdf in pdf_files:
    txt_path = TXT_DIR / pdf.with_suffix(".txt").name
    if txt_path.exists():
        print(f"✓ {txt_path.name} exists")
        ok += 1
        continue

    print(f"→ extracting {pdf.name} … ", end="", flush=True)
    try:
        text = extract_text(str(pdf))
        txt_path.write_text(text, encoding="utf-8")
        print("done")
        ok += 1
    except Exception as exc:
        print("FAILED")
        print(f"   !! {pdf.name}: {exc}", file=sys.stderr)
        fail += 1

print(f"\nFinished – {ok} extracted, {fail} failed.")


