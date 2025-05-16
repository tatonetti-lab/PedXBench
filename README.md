
# PED-X-Bench: FDA Pediatric Drug Extrapolation Dataset

This repository contains the dataset, benchmark tasks, and baseline models for our NeurIPS 2025 submission:

> **PED-X-Bench: A Dataset for Modeling FDA Pediatric Drug Extrapolation Decisions**

## 🧾 Overview

**PED-X-Bench** is a benchmark for evaluating models on the task of predicting whether the U.S. FDA extrapolated adult drug data to children in labeling decisions. It includes:

- ✅ 778 structured FDA drug label entries (1988–2024)
- ✅ Extrapolation labels: `Full`, `Partial`, `None`, `Unlabeled`
- ✅ Summaries of pediatric efficacy and PK/safety evidence
- ✅ Annotated rationales and pediatric study characteristics
- ✅ Manually adjudicated subset of 135 entries

The full dataset (Parquet + Croissant metadata + raw text) is **hosted on
Hugging Face**:

> <https://huggingface.co/datasets/apoorvasrinivasan/Ped-X-Bench>

```bash
# 1. Install HF datasets helpers if you haven’t.
pip install "datasets>=2.18" huggingface_hub

# 2. Pull the Parquet files (splits + metadata) into ./data/hf_cache/
python - <<'PY'
from datasets import load_dataset
load_dataset("apoorvasrinivasan/Ped-X-Bench", data_dir=".", split="train")
PY
```

🔻 Optional: download the raw .txt FDA labels locally

```
python scripts/download_txt.py \
        --hf-dataset apoorvasrinivasan/Ped-X-Bench \
        --dst data/raw/txt
```

This creates the exact directory layout expected by train_bigbird.py.

## Quick-start: reproduce the BigBird baseline

# 1. Create a clean environment
conda create -n pedx-bench python=3.10 -y
conda activate pedx-bench
pip install -r requirements.txt          # transformers[torch], datasets, accelerate, evaluate, scikit-learn, sentencepiece

# 2. Train for a single epoch (≈20 min on 1 × A100; CPU works but is slower)
```
python scripts/train_bigbird.py \
       --split_dir data/processed/splits \
       --txt_dir   data/raw/txt \
       --out_dir   checkpoints/bigbird_demo \
       --epochs    4
```
The script prints dev metrics every 100 steps and writes:
checkpoints/bigbird_demo/
  ├── config.json
  ├── pytorch_model.bin
  ├── tokenizer.json
  └── test_metrics.json

## Evaluate the saved model

python scripts/eval_bigbird.py \
       --model_dir checkpoints/bigbird_demo \
       --split_csv data/processed/splits/test.csv \
       --txt_dir   data/raw/txt

      
