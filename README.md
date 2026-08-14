# Banking77 Intent Classification

A complete NLP pipeline for classifying banking customer queries into **77 intents** using **DistilBERT + PyTorch Lightning**.

## Results

| Metric            |      Score |
| ----------------- | ---------: |
| **Test Accuracy** | **93.05%** |
| **Macro F1**      | **93.05%** |
| Test Loss         |  **0.299** |

## Pipeline

```text
Banking77
   ↓
EDA
   ↓
Label Encoding
   ↓
DistilBERT Tokenizer
   ↓
Dataset
   ↓
Train / Validation Split
   ↓
Dynamic Padding
   ↓
DataLoader
   ↓
LightningDataModule
   ↓
DistilBERT
   ↓
AdamW
   ↓
Warmup + Cosine Scheduler
   ↓
Mixed Precision
   ↓
ModelCheckpoint
   ↓
EarlyStopping
   ↓
Evaluation
   ↓
Inference
```

## Highlights

* 77-class banking intent classification
* DistilBERT fine-tuning
* Dynamic padding with `DataCollatorWithPadding`
* Stratified train/validation split
* AdamW optimizer
* Warmup + cosine learning-rate schedule
* Mixed precision training
* Early stopping and best-checkpoint selection
* Macro Precision / Recall / F1
* Confusion matrix and per-class analysis

## Project Structure

```text
Bannking_77/
│
├── img/
│
├── project/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── evaluation.py
│   ├── Eda.ipynb
│   └── evaluation.ipynb


```

## Training Configuration

```text
Model            : DistilBERT
Classes          : 77
Batch Size       : 32
Learning Rate    : 1e-4
Weight Decay     : 0.01
Epochs           : 8
Warmup           : 10%
Precision        : 16-bit Mixed Precision
Gradient Clipping: 1.0
```

## Error Analysis

The model performs strongly overall, while most difficult cases come from **semantically similar intents**.

Examples:

```text
why_verify_identity
        ↕
verify_my_identity
```

```text
virtual_card_not_working
        ↕
get_disposable_virtual_card
```

This was explored using per-class metrics and a confusion matrix.

## Trained Model and dataset

The best trained checkpoint is available here:

**[Google Drive — Trained Model](https://drive.google.com/drive/folders/120QOqrYP1vfdPYBx9LFMMKZjOPTlz_a4?usp=sharing)**

## Tech Stack

`Python` · `PyTorch` · `PyTorch Lightning` · `Hugging Face Transformers` · `DistilBERT` · `TorchMetrics` · `scikit-learn`

## What's Next

* Confidence-based inference
* Deeper error analysis
* Further fine-tuning
* Comparison with larger transformer architectures

---

**Built as an end-to-end NLP project — from EDA to training, evaluation, and inference.**
