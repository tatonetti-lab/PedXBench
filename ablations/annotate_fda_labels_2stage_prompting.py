#!/usr/bin/env python3
"""
annotate_fda_labels_fc.py  –  two-stage Azure-OpenAI **function-calling** pipeline
==========================================================================

)
"""

import os, json, time, traceback
from pathlib import Path
import pandas as pd
from openai import AzureOpenAI                    # Azure & public client share the same v1 SDK
from dotenv import load_dotenv   # pip install python-dotenv
import os
# ────────────────────────────────────────────  CONFIG


load_dotenv()


AZURE_OPENAI_ENDPOINT   = os.environ("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY       = os.environ("AZURE_OPENAI_KEY")
API_VERSION      = os.environ("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME  = "o3-mini"



# ──────────────────────────────────────────────────────────────────────────
# Azure client
client = AzureOpenAI(
    api_key        = AZURE_OPENAI_KEY,
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    api_version    = API_VERSION
)


# ──────────────────────────────────────────────────────────────────────────
# folders / output
TXT_DIR  = Path("/Users/srinivasana/Documents/peds_agents_local/txt")
OUT_CSV  = TXT_DIR.parent / "llm_2stage_fc_predictions_full.csv"
ERR_CSV  = TXT_DIR.parent / "llm_2stage_fc_errors_full.csv"

# ──────────────────────────────────────────────────────────────────────────
# 1️⃣  function-call declaration : Stage 1  (extract)
extract_func = {
    "name": "extract_pediatric_summary",
    "description": (
        "Scan an FDA product label and return every piece of pediatric evidence "
        "needed to judge extrapolation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "PediatricSummary": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section": { "type": "string" },
                        "summary": { "type": "string" }
                    },
                    "required": ["section","summary"]
                }
            },
            "AllAges" : { "type": "array", "items": { "type": "string" } },
            "Comments": { "type": "string" }
        },
        "required": ["PediatricSummary","AllAges","Comments"]
    },
}

EXTRACT_SYSTEM = (
    "You are scanning an FDA product label. "
    "Return JSON ONLY via the function call. Follow this schema exactly. "
    "• Summaries ≤150 words. • DO NOT invent data."
)

# ──────────────────────────────────────────────────────────────────────────
# 2️⃣  function-call declaration : Stage 2  (classify)
classify_func = {
    "name": "classify_extrapolation",
    "description": "Given the pediatric summary, decide extrapolation category.",
    "parameters": {
        "type": "object",
        "properties": {
            "resolved_label": {
                "type":"string",
                "enum":["None","Partial","Full","Unlabeled"]},
            "peds_study_type": {
                "type":"string",
                "enum":["RCT","PK+Safety","PK Only","None"]},
            "efficacy_summary": { "type":"string" },
            "pk_summary"     : { "type":"string" },
            "lowest_age_band": { "type":"string" },
            "highest_age_band":{ "type":"string" },
            "rationale"      : { "type":"string" },
            "confidence"     : {
                "type":"string",
                "enum":["high","medium","low"]},
        },
        "required":[
            "resolved_label","peds_study_type","efficacy_summary","pk_summary",
            "lowest_age_band","highest_age_band","rationale","confidence"
        ]
    },
}

CLASSIFY_SYSTEM = (
    "You are an expert in FDA pediatric extrapolation. "
    "Use the decision tree:\n"
    "• None – ≥1 pediatric efficacy RCT\n"
    "• Partial – pediatric PK and/or safety evidence but NO efficacy RCT\n"
    "• Full – only PK / exposure modelling; no pediatric safety cohort\n"
    "• Unlabeled – no pediatric evidence\n\n"
    "Return JSON ONLY via the function call."
)

# ──────────────────────────────────────────────────────────────────────────
def ask_function(messages, functions, force_name):
    """Single call that returns the *arguments* dict for the function."""
    resp = client.chat.completions.create(
        model         = DEPLOYMENT_NAME,
        messages      = messages,
        functions     = functions,
        function_call = {"name": force_name},   # enforce call
        max_completion_tokens    = 32000,
    )
    call = resp.choices[0].message.function_call
    return json.loads(call.arguments)

# ──────────────────────────────────────────────────────────────────────────
records, errors = [], []

for txt_file in sorted(TXT_DIR.glob("*.txt")):
    app_id = txt_file.stem
    txt    = txt_file.read_text("utf-8", errors="ignore")

    try:
        # -------- Stage 1
        stage1_args = ask_function(
            messages=[
                {"role":"system","content":EXTRACT_SYSTEM},
                {"role":"user"  ,"content":txt}
            ],
            functions=[extract_func],
            force_name="extract_pediatric_summary"
        )

        # -------- Stage 2
        stage2_args = ask_function(
            messages=[
                {"role":"system","content":CLASSIFY_SYSTEM},
                {"role":"user"  ,"content":json.dumps(stage1_args, separators=(',',':'))}
            ],
            functions=[classify_func],
            force_name="classify_extrapolation"
        )

        # -------- record
        records.append({
            "app_id"         : app_id,
            **stage2_args,
            "summary_json"   : json.dumps(stage1_args, separators=(',',':')),
            "txt_file"       : str(txt_file)
        })
        print(f"✓ {app_id:>14} → {stage2_args['resolved_label']:9} ({stage2_args['confidence']})")

    except Exception as exc:
        traceback.print_exc(limit=1)
        errors.append({"app_id":app_id,"error":str(exc)})
        print(f"✗ {app_id} FAILED – {exc}")

    time.sleep(0.3)    # gentle throttle – adjust as needed

# ──────────────────────────────────────────────────────────────────────────
if records:
    pd.DataFrame(records).to_csv(OUT_CSV, index=False)
    print(f"\n✔ Saved {len(records)} rows → {OUT_CSV}")

pd.DataFrame(errors).to_csv(ERR_CSV, index=False)
print(f"Finished. {len(records)} OK, {len(errors)} errors → {ERR_CSV}")
