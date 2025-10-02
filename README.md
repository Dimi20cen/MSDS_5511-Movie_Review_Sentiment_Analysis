# IMDb Movie Review Sentiment Analysis

A compact notebook project that classifies IMDb reviews as **positive** or **negative**. It builds a fast **1D CNN** baseline and then fine‑tunes **DistilBERT** for higher accuracy.

## Results (Test Set)

- **DistilBERT:** 91.4% accuracy · Precision 0.95 · Recall 0.87 · F1 0.91
- **1D CNN:** 88.6% accuracy · loss 0.28

## Approach

1. **EDA & Prep:** HTML strip, lowercase, review length analysis.
2. **Baseline (CNN):** Tokenizer → pad to 500 → Embedding → Conv1D → GlobalMaxPool.
3. **Transformer (DistilBERT):** `distilbert-base-uncased` fine‑tuned with dynamic padding `DataCollatorWithPadding`.
4. **Evaluation:** accuracy, confusion matrix, classification report.
