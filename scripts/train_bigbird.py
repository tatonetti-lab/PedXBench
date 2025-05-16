#!/usr/bin/env python
"""
Reproducible BigBird baseline.

Example
-------
python scripts/train_bigbird.py \\
  --split_dir data/splits \\
  --txt_dir   data/txt   \\
  --out_dir   checkpoints/bigbird \\
  --epochs    4
"""
from pathlib import Path
import argparse, json, numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import (
    BigBirdTokenizerFast, BigBirdForSequenceClassification,
    TrainingArguments, Trainer, DefaultDataCollator
)
import evaluate

LABEL2ID = {"NotExtrapolated": 0, "Partial": 1, "Full": 2, "Unlabeled": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------- helpers ---------------------------------------------------- #
def _canon_txt(txt_dir: Path, cid: str) -> Path:
    return txt_dir / f"{cid}.txt"

def _filter_ok(df: pd.DataFrame, txt_dir: Path) -> pd.DataFrame:
    df = df.dropna(subset=["canon_id"]).copy()
    df["txt_file"] = df["canon_id"].apply(lambda c: _canon_txt(txt_dir, c))
    df = df[df["txt_file"].apply(Path.exists)]
    df["label"] = df["label"].replace("", "NotExtrapolated").map(LABEL2ID).astype(int)

    df["txt_file"] = df["txt_file"].astype(str)      # <-- add this cast
    assert len(df), "No rows left after matching txt files – check paths!"
    return df[["txt_file", "label"]]


def load_splits(split_dir: Path, txt_dir: Path):
    dfs = [pd.read_csv(split_dir / f"{s}.csv", dtype=str) for s in ["train","dev","test"]]
    return [Dataset.from_pandas(_filter_ok(df, txt_dir)) for df in dfs]

# ---------- main ------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, type=Path)
    ap.add_argument("--txt_dir",   required=True, type=Path)
    ap.add_argument("--out_dir",   default=Path("checkpoints/bigbird"), type=Path)
    ap.add_argument("--epochs",    default=4, type=int)
    args = ap.parse_args()

    ds_train, ds_dev, ds_test = load_splits(args.split_dir, args.txt_dir)

    tok = BigBirdTokenizerFast.from_pretrained("google/bigbird-roberta-base")
    def tok_fn(batch):
        texts = [Path(p).read_text(errors="ignore") for p in batch["txt_file"]]
        enc   = tok(texts, max_length=4096, truncation=True, padding="max_length")
        enc["labels"] = batch["label"]
        return enc

    ds_train = ds_train.map(tok_fn, batched=True, remove_columns=["txt_file","label"]).with_format("torch")
    ds_dev   = ds_dev  .map(tok_fn, batched=True, remove_columns=["txt_file","label"]).with_format("torch")
    ds_test  = ds_test .map(tok_fn, batched=True, remove_columns=["txt_file","label"]).with_format("torch")

    model = BigBirdForSequenceClassification.from_pretrained(
        "google/bigbird-roberta-base",
        num_labels=4, label2id=LABEL2ID, id2label=ID2LABEL,
        gradient_checkpointing=True)

    metrics = {"accuracy": evaluate.load("accuracy"),
               "f1":       evaluate.load("f1")}

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": metrics["accuracy"](preds, labels)["accuracy"],
                "f1":       metrics["f1"](preds, labels, average="macro")["f1"]}

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=True,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        seed=42,
        eval_strategy="steps",            # transformers>=4.45
        eval_steps=100,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        data_collator=DefaultDataCollator(return_tensors="pt"),
        compute_metrics=compute)

    trainer.train()
    metrics_test = trainer.evaluate(ds_test, metric_key_prefix="test")
    print(json.dumps(metrics_test, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    (args.out_dir / "test_metrics.json").write_text(json.dumps(metrics_test, indent=2))

if __name__ == "__main__":
    main()

