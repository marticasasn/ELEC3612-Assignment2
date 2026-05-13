# Music Genre Classification — ELEC3612 Assignment 2

**Deep Learning for Audio Understanding**  
University of Sydney — Pattern Recognition and Machine Intelligence  
Group Assignment — 2026

---

## Overview

This project implements a deep learning pipeline for automatic music genre classification using the [lewtun/music_genres](https://huggingface.co/datasets/lewtun/music_genres) dataset. After filtering 8 ambiguous or under-represented genres, the working dataset contains **17,632 audio clips** across **12 genres**.

We develop and compare two deep learning architectures against a classical ML baseline:

- **CNN 2D** — trained on mel-spectrograms (audio treated as an image)
- **BiLSTM** — trained on MFCC sequences (audio treated as a time series)
- **SVM baseline** — RBF kernel on pooled MFCC features (mean + std per frame)

---

## Dataset

### Raw dataset

| Property | Value |
|---|---|
| Source | HuggingFace — `lewtun/music_genres` |
| Total samples (raw) | 24,985 |
| Clip duration | 30 seconds |
| Sampling rate | 22,050 Hz |
| Total size | ~10 GB |

### After filtering (12 genres)

8 genres removed: Unknown, Easy Listening, Blues, Soul-RnB, Spoken, Old-Time / Historic, Ambient Electronic, International.

| Split | Samples |
|---|---|
| Train | 12,342 |
| Validation | 2,645 |
| Test | 2,645 |
| **Total** | **17,632** |

**Genres:** Chiptune / Glitch, Classical, Country, Electronic, Experimental, Folk, Hip-Hop, Instrumental, Jazz, Pop, Punk, Rock.

---

## Preprocessed Dataset (Kaggle)

To avoid re-running the full 10 GB preprocessing pipeline, we publish pre-extracted tensors on Kaggle:

**Kaggle Dataset:** [[Kaggle preprocessed dataset link]](https://www.kaggle.com/datasets/marticasas/pr-a2-preprocesseddataset)

| File | Shape | Description |
|---|---|---|
| `spectrograms_{train,val,test}.pt` | (N, 1, 128, 128) | Normalised mel-spectrograms |
| `mfccs_{train,val,test}.pt` | (N, 130, 40) | Normalised MFCC sequences |
| `labels_{train,val,test}.pt` | (N,) | Integer class labels |
| `genre_mapping.json` | — | 12 genres, 0-indexed |
| `spectrogram_stats.json` | — | Mean/std computed on train split only |
| `mfcc_stats.json` | — | Mean/std computed on train split only |

**Total size:** ~1.45 GB

---

## Project Structure

```
ELEC3612-Assignment2/
│
├── notebooks/
│   ├── 01_eda_kaggle.ipynb          # Exploratory data analysis and genre filtering
│   ├── 02_preprocessing_kaggle.ipynb  # Feature extraction and tensor export
│   ├── 03_cnn_kaggle.ipynb          # CNN 2D training on mel-spectrograms
│   ├── 04_lstm_kaggle.ipynb         # BiLSTM training on MFCC sequences
│   └── 05_evaluation_kaggle.ipynb   # Comparative evaluation: CNN vs BiLSTM vs SVM
│
├── src/
│   ├── dataset.py     # SpectrogramDataset and MFCCDataset (PyTorch Dataset classes)
│   ├── models.py      # CNNClassifier and LSTMClassifier architectures
│   ├── train.py       # Training loop with early stopping and checkpointing
│   └── evaluate.py    # Metrics, confusion matrix, and training curve plots
│
├── results/
│   └── figures/       # EDA plots and confusion matrices
│
├── report/            # Final technical report
├── docs/              # Project roadmap
└── requirements.txt
```

---

## Pipelines

### Pipeline A — Mel-Spectrogram + CNN 2D

Each 30-second clip is converted to a log-scaled mel-spectrogram, resized to 128×128, and classified with a 4-block Conv2D network.

- **Input shape:** (N, 1, 128, 128)
- **Architecture:** 4× [Conv2d → BatchNorm → ReLU → MaxPool] → AdaptiveAvgPool → Dropout → Linear(512) → Linear(12)
- **Strength:** captures local spectral texture, timbre, and rhythm

### Pipeline B — MFCCs + BiLSTM

Each clip is represented as a sequence of 130 MFCC frames and fed into a 2-layer bidirectional LSTM.

- **Input shape:** (N, 130, 40)
- **Architecture:** BiLSTM(hidden=256, layers=2) → concat final hidden states → Dropout → Linear(256) → Linear(12)
- **Strength:** captures how musical structure evolves over time

### Baseline — SVM on pooled MFCCs

Per-frame MFCC mean and std are concatenated to produce an 80-dimensional feature vector, scaled with StandardScaler, and classified with an RBF-kernel SVM.

---

## Notebook Execution Order

All notebooks are designed to run on **Kaggle** (free GPU, 30h/week). Run them in this order:

| Step | Notebook | Description |
|---|---|---|
| 1 | `01_eda_kaggle.ipynb` | EDA, genre filtering, visualisations |
| 2 | `02_preprocessing_kaggle.ipynb` | Extract spectrograms and MFCCs, export tensors |
| 3 | `03_cnn_kaggle.ipynb` | Train CNN, save checkpoint and predictions |
| 4 | `04_lstm_kaggle.ipynb` | Train BiLSTM, save checkpoint and predictions |
| 5 | `05_evaluation_kaggle.ipynb` | Compare all models, generate final figures |

### How to add the preprocessed dataset on Kaggle

1. Open any of notebooks 03–05 on Kaggle.
2. Click **+ Add Data** → search for `pr-a2-preprocesseddataset` by `marticasas`.
3. The data will be available at `/kaggle/input/pr-a2-preprocesseddataset/`.

For notebook 05, also add CNN and LSTM results as separate datasets:
- `cnn-results` — [TODO: CNN results dataset link] — containing `best_cnn.pth` and `cnn_predictions.csv`
- `lstm-results` — [TODO: LSTM results dataset link] — containing `best_lstm.pth` and `lstm_predictions.csv`

---

## src/ Modules

| Module | Description |
|---|---|
| `dataset.py` | `SpectrogramDataset` and `MFCCDataset` — load `.pt` tensors, validate shapes, compute class weights |
| `models.py` | `CNNClassifier` and `LSTMClassifier` — architecture definitions, parameter count |
| `train.py` | `train()` loop with gradient clipping, early stopping, LR scheduling, and checkpointing |
| `evaluate.py` | `compute_metrics()`, `plot_confusion_matrix()`, `plot_training_curves()`, `evaluate_model()` |

---

## Setup (local)

```bash
pip install -r requirements.txt
```

> Training is intended to run on Kaggle. Local setup is only needed for development or EDA.

---

## Evaluation Metrics

- Accuracy
- Weighted F1-score
- Macro F1-score
- Per-class F1-score
- Confusion matrix (row-normalised)
- Training loss and accuracy curves

---

## Course

**ELEC3612 — Pattern Recognition and Machine Intelligence**  
School of Electrical and Information Engineering  
University of Sydney, 2026
