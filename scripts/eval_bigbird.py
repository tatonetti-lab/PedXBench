#!/usr/bin/env python
"""
Evaluate a fine-tuned BigBird model on a given split.

Example
-------
python scripts/eval_bigbird.py \
    --model_dir checkpoints/bigbird_full \
    --split_csv data/processed/splits/test.csv \
    --txt_dir   data/raw/txt
"""
import argparse, json, numpy as np, pandas as pd, torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    BigBirdTokenizerFast,
    BigBirdForSequenceClassification,
    Trainer,
    DefaultDataCollator,
)

LABEL2ID = {"NotExtrapolated": 0, "Partial": 1, "Full": 2, "Unlabeled": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# --------------------------------------------------------------------------- #
def load_split(csv_path: Path, txt_dir: Path, tok: BigBirdTokenizerFast) -> Dataset:
    """Return a tokenised HF Dataset ready for evaluation."""
    df = pd.read_csv(csv_path, dtype=str)
    # map canon_id → raw text
    df["text"] = df["canon_id"].apply(
        lambda cid: (txt_dir / f"{cid}.txt").read_text(errors="ignore")
    )
    df["label_id"] = df["label"].replace("", "NotExtrapolated").map(LABEL2ID).astype(int)

    ds = Dataset.from_pandas(df[["text", "label_id"]])

    def tok_fn(batch):
        enc = tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=4096,
        )
        enc["labels"] = batch["label_id"]
        return enc

    ds = ds.map(tok_fn, batched=True, remove_columns=["text", "label_id"])
    ds.set_format("torch")
    return ds


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=Path, help="Checkpoint folder")
    ap.add_argument("--split_csv", required=True, type=Path, help="CSV (test/dev)")
    ap.add_argument("--txt_dir", required=True, type=Path, help="Folder with .txt labels")
    ap.add_argument("--out_json", type=Path, help="Optional: write metrics json here")
    args = ap.parse_args()

    # load model + matching tokenizer
    tok = BigBirdTokenizerFast.from_pretrained(args.model_dir)
    model = BigBirdForSequenceClassification.from_pretrained(args.model_dir)

    ds = load_split(args.split_csv, args.txt_dir, tok)

    trainer = Trainer(
        model=model,
        data_collator=DefaultDataCollator(return_tensors="pt"),
    )
    preds = trainer.predict(ds)
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(1)

    accuracy = (y_pred == y_true).mean().item()
    metrics = {"accuracy": accuracy}

    print(json.dumps(metrics, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
