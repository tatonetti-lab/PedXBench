import os, json, re, time, traceback, textwrap, math
from pathlib import Path
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv

import os
# ────────────────────────────────────────────  CONFIG


load_dotenv()


AZURE_OPENAI_ENDPOINT   = os.environ("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY       = os.environ("AZURE_OPENAI_KEY")
API_VERSION      = os.environ("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME  = "o3-mini"


client = AzureOpenAI(
    api_key        = AZURE_OPENAI_KEY,
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    api_version    = API_VERSION
)

# ───── Paths ──────────────────────────────────────────────────────────────────
TXT_DIR  = Path("/Users/srinivasana/Documents/peds_agents/txt")
OUT_OK   = TXT_DIR.parent / "llm_predictions_o3-mini_v4_updated_full.csv"
OUT_ERR  = TXT_DIR.parent / "llm_errors_o3-mini_v4_updated_full.log"

# ───── Prompt ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert in FDA & ICH-E11A pediatric extrapolation.
Return ONLY a JSON object with these keys:

  resolved_label      : "None" | "Partial" | "Full" | "Unlabeled"
  peds_study_type     : "RCT" | "PK+Safety" | "PK Only" | "None"
  efficacy_excerpt    : ≤ 2 verbatim sentences proving pediatric efficacy ("" if none)
  pk_excerpt          : ≤ 2 sentences showing pediatric PK / dose bridging ("" if none)
  lowest_age_band     : youngest pediatric age explicitly mentioned, e.g. "3 months" or ""
  highest_age_band :    oldest pediatric age explicitly mentioned, e.g. "16 years" or ""
  rationale           : ≤ 50-word sentence explaining your decision
  confidence          : "high" | "medium" | "low"


☆  LABEL TAXONOMY ─────────────────────────────────────────────────────────────
None      – pediatric randomised, adequate & well-controlled clinical endpoint study.
Partial   – adult efficacy + pediatric PK and/or safety bridge only.
Full      – pediatric PK-only bridge; safety and efficacy fully extrapolated from adults.
Unlabeled – label says “Safety and effectiveness … not established” and has no paediatric PK.

☆  QUICK RULE-OF-THUMB
None  = pediatric RCT proves efficacy.  
Partial = adult RCTs + pediatric PK/safety bridge.  
Full  = pediatric PK-only bridge.  
Unlabeled = nothing for kids.

☆  Tips & traps
      • Age ≥ 17 y only = adult → ignore.
      • “Clinical trial” alone ≠ RCT. Look for “randomised”, “double-blind”,
        “adequate & well-controlled”.
    • Paediatric arm embedded in an adult RCT:
         – If randomised, well powered & analysed for *clinical efficacy* → None.  
         – If safety / PK only → Partial.  
    • Open-label or single-arm safety cohorts → Partial.
    • Mixed indications: if *any* indication is Partial, classify whole label as Partial.
    • Use your best judgement. If you cannot decide, choose the *more conservative* category and lower
        confidence to “medium” or “low”.



Output MUST be valid JSON, no markdown fences, no extra keys.
--- END OF INSTRUCTIONS ---
""").strip()

# ───── Helper: slide long labels into context windows ─────────────────────────
CHARS_PER_WINDOW = 6_000        # plenty of room with the new ~900-token prompt

def sliding_windows(text: str, size: int = CHARS_PER_WINDOW):
    step = size - 20            # 20-char overlap
    for i in range(0, len(text), step):
        yield text[i : i + size]

# ───── LLM wrapper ────────────────────────────────────────────────────────────
def label_with_llm(label_txt: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for n, chunk in enumerate(sliding_windows(label_txt)):
        prefix = f"[PART {n+1}/{math.ceil(len(label_txt)/CHARS_PER_WINDOW)}]\n"
        messages.append({"role": "user", "content": prefix + chunk})

    messages.append({"role": "user", "content": "End of label. Provide the JSON object now."})

    resp = client.chat.completions.create(
        model    = DEPLOYMENT_NAME,
        messages = messages,
        max_completion_tokens = 32000,
    )

    raw = resp.choices[0].message.content.strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError(f"No JSON detected in:\n{raw[:300]}")

    return json.loads(m.group())

# ───── Main loop ──────────────────────────────────────────────────────────────
ALL_COLS = [
    "app_id", "resolved_label", "peds_study_type",
    "efficacy_excerpt", "pk_excerpt", "lowest_age_band",
    "highest_age_band",
    "rationale", "confidence",
    # optional extras — keep stable order
    # "indications_covered", "pk_bridge_method", "num_pediatric_subjects",
    # "endpoint_type", "study_design_detail", "evidence_page_refs",
    "ambiguity_flag", "notes", "txt_file"
]

records, errors = [], []

for txt_path in sorted(TXT_DIR.glob("*.txt")):
    app_id = txt_path.stem
    try:
        txt = txt_path.read_text("utf-8", errors="ignore")
        result = label_with_llm(txt)

        # Fill any missing optional keys with blanks so CSV schema is stable
        for k in ALL_COLS:
            if k not in result:
                result[k] = "" if k != "ambiguity_flag" else False

        result.update({"app_id": app_id, "txt_file": str(txt_path)})
        records.append(result)

        print(f"✓ {app_id:>7} → {result['resolved_label']:8} ({result['confidence']})")
        time.sleep(0.3)

    except Exception as exc:
        traceback.print_exc(limit=2)
        errors.append({"app_id": app_id, "error": str(exc)})
        print(f"✗ {app_id} FAILED")

# ───── Persist ────────────────────────────────────────────────────────────────
pd.DataFrame(records)[ALL_COLS].to_csv(OUT_OK, index=False)
pd.DataFrame(errors ).to_csv(OUT_ERR, index=False)

print(f"\nSaved {len(records)} predictions ➜ {OUT_OK}")
print(f"Errors: {len(errors)} (see {OUT_ERR})")

