# Pain Classification 

Multivariate time-series classification challenge at Politecnico di Milano.  
The task is to classify each pirate's perceived pain level as `no_pain`, `low_pain`, or `high_pain` based on body-joint-angle motion-capture sequences. Success is measured by **weighted F1-score**.

---

## Problem Description

| Item | Detail |
|------|--------|
| Training samples | 661 pirates |
| Test samples | 1 324 pirates |
| Timesteps per sample | 160 |
| Features per timestep | 40 |
| Classes | `no_pain` (77 %), `low_pain` (14 %), `high_pain` (8 %) |
| Metric | Weighted F1-score |

### Feature groups

| Group | Columns | Description |
|-------|---------|-------------|
| Temporal index | `time` | Timestep 0–159 |
| Pain surveys | `pain_survey_1–4` | Self-reported pain (integer 0–2) |
| Morphology | `n_legs`, `n_hands`, `n_eyes` | Categorical pirate traits |
| Joint angles | `joint_00–joint_29` | Real-valued body-joint angles |
| Constant | `joint_30` | Zero-variance column (dropped) |

---

## Repository Structure

```
Time-Series-Classification/
├── README.md
├── AN2DL_first_Report.pdf        ← Full project report
├── .gitignore
└── notebooks/
    ├── bidirlstm_1dconv.ipynb    ← ⭐ BEST MODEL (val F1 ≈ 0.935)
    ├── cnn_lstm_attention.ipynb  ← Experiment: CNN + BiLSTM + Attention (val F1 ≈ 0.924)
    ├── conv1d_ann.ipynb          ← Experiment: Conv1D (time-wise) + BiLSTM (val F1 ≈ 0.927)
    └── pca_ann.ipynb             ← Baseline: PCA + BiLSTM (val F1 ≈ 0.708)
```

All notebooks are self-contained and designed to run on **Kaggle** (GPU available, data at `/kaggle/input/pirate-pain/`).

---

## Best Model — BiDirectional LSTM + 1D Conv

### Architecture

```
Input branches:
  Joint angles (30) ──► Conv1D(30→16, kernel=3, same-padding) ──►
  n_legs integer    ──► Embedding(vocab=2, dim=3)               ──► Concat(16+3+4 = 23)
  Pain surveys (4)  ──────────────────────────────────────────────►
                         BiLSTM(hidden=128, layers=2, dropout=0.2)
                              ↓ last hidden state (fwd ‖ bwd = 256)
                         Linear(256 → 3) + LogSoftmax
```

The **feature-wise Conv1D** scans across the 30 joint channels at each timestep, extracting cross-joint local patterns before passing the result to the Bidirectional LSTM.

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| **Feature-wise Conv1D** | Learns cross-joint spatial correlations before the LSTM processes time |
| **Bidirectional LSTM** | Captures both preceding and following motion context |
| **`nn.Embedding` for morphology** | Treats pirate body type as a learned representation, not a raw integer |
| **Weighted CrossEntropyLoss** | Compensates for severe class imbalance; `weight ∝ 1 / class_count` |
| **Majority voting** | Each pirate's label is the mode prediction across all overlapping windows |
| **Early stopping (patience=50)** | Prevents overfitting; best checkpoint is restored automatically |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Window size | 20 |
| Stride | 5 |
| Conv1D channels | 16 |
| LSTM hidden size | 128 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| L2 weight decay | 1e-3 |
| Batch size | 512 |
| Max epochs | 512 |
| Early stopping patience | 50 |

### Results

| Split | Weighted F1 |
|-------|-------------|
| Validation | **0.9352** |

Training converged in ~200 epochs with early stopping restoring the epoch-150 checkpoint.

---

## Experiments Summary

| Notebook | Model | Val F1 | Key idea |
|----------|-------|--------|----------|
| `bidirlstm_1dconv.ipynb` | **BiDir LSTM + Conv1D** | **0.935** | Feature-wise conv extracts cross-joint patterns |
| `conv1d_ann.ipynb` | Conv1D (time-wise) + BiLSTM | 0.927 | Time-axis conv before LSTM |
| `cnn_lstm_attention.ipynb` | CNN + BiLSTM + Attention | 0.924 | Tanh attention over LSTM states; thresholding of sparse joints |
| `pca_ann.ipynb` | PCA + BiDir LSTM | 0.708 | PCA compresses joints to 6 PCs — loses temporal structure |

### What helped

- **Weighted loss** — the single biggest gain; without it the model predicts `no_pain` for almost everything
- **Majority voting** — aggregating window-level predictions at the sample level reduces noise
- **Embedding layer** — representing pirate morphology as a dense vector outperforms treating it as a scalar
- **Bidirectionality** — capturing future context inside each window helps the LSTM

### What did not help

- **PCA** — dimensionality reduction destroys the joint-angle structure the LSTM depends on
- **L1 regularization** — added no benefit over L2 alone
- **Attention** — minor improvement over taking the final hidden state; not worth the added complexity

---

## Data

Data is hosted on Kaggle and is **not included** in this repository.

To run the notebooks:
1. Create a Kaggle account and accept the competition data terms
2. Download the dataset to `/kaggle/input/pirate-pain/` (done automatically on Kaggle)
3. Run any notebook in a Kaggle kernel with GPU acceleration enabled

Expected files:
```
/kaggle/input/pirate-pain/
├── pirate_pain_train.csv        (105 760 rows × 40 columns)
├── pirate_pain_train_labels.csv (661 rows × 2 columns)
└── pirate_pain_test.csv         (211 840 rows × 40 columns)
```

---

## Report

The full project report is available in [`AN2DL_first_Report.pdf`](AN2DL_first_Report.pdf).  
It covers the problem formulation, data analysis, all model architectures, ablation study, and final results.
