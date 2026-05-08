# Music Genre Classification — ELEC3612 Assignment 2

**Deep Learning for Audio Understanding**  
University of Sydney — Pattern Recognition and Machine Intelligence  
Group Assignment — 2025

---

## Overview

This project implements a deep learning pipeline for automatic music genre classification using the [lewtun/music_genres](https://huggingface.co/datasets/lewtun/music_genres) dataset (~25,000 audio clips, 30 seconds each, across 19 genres).

We develop and compare two deep learning architectures:

- **CNN 2D** — trained on mel-spectrograms (audio treated as an image)
- **LSTM** — trained on MFCC sequences (audio treated as a time series)

Results are evaluated against a classical ML baseline (SVM / Random Forest on hand-crafted features).

---

## Dataset

| Property | Value |
|---|---|
| Source | HuggingFace — `lewtun/music_genres` |
| Total samples | ~24,985 |
| Train / Test split | 19,905 / 5,080 |
| Clip duration | 30 seconds |
| Sampling rate | 22,050 Hz |
| Number of genres | 19 (subject to filtering after EDA) |
| Total size | ~10 GB |

**Genres:** Electronic, Rock, Hip-Hop, Folk, Experimental, Punk, Pop, Jazz, Classical, Instrumental, Ambient Electronic, International, Chiptune/Glitch, Country, Blues, Old-Time/Historic, Spoken, and others.

> Note: some genres may be removed after exploratory data analysis based on class imbalance or acoustic ambiguity (e.g. "Spoken", "Old-Time / Historic").

---

## Project Structure

```
music-genre-classification/
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb     # Feature extraction (spectrograms, MFCCs)
│   ├── 03_cnn.ipynb                # Model A: CNN 2D on mel-spectrograms
│   └── 04_lstm.ipynb               # Model B: LSTM on MFCCs
│
├── src/
│   ├── dataset.py                  # PyTorch Dataset classes
│   ├── models.py                   # Model architectures
│   ├── train.py                    # Training loop with checkpointing
│   └── evaluate.py                 # Metrics and visualisations
│
├── results/
│   └── figures/                    # Confusion matrices, loss curves, etc.
│
├── report/                         # Final technical report
└── README.md
```

---

## Pipelines

### Pipeline A — Mel-Spectrogram + CNN 2D

Each 30-second audio clip is converted into a mel-spectrogram — a 2D time-frequency representation — which is then treated as an image and classified using a Convolutional Neural Network.

- **Input:** mel-spectrogram (shape: 1 × 128 × 128)
- **Architecture:** Conv2D → BatchNorm → ReLU → MaxPool → Dropout → Linear
- **Strength:** captures local spectral patterns (texture, timbre, rhythm)

### Pipeline B — MFCCs + LSTM

Each clip is represented as a sequence of MFCC frames and fed into a Long Short-Term Memory network, which learns temporal patterns across the duration of the song.

- **Input:** MFCCs (shape: ~130 timesteps × 40 features)
- **Architecture:** Stacked LSTM → Dropout → Linear classifier
- **Strength:** captures how music evolves over time

---

## Setup

### Requirements

```bash
pip install datasets torch torchaudio librosa matplotlib seaborn scikit-learn numpy pandas
```

### Load the dataset

```python
from datasets import load_dataset, concatenate_datasets

ds = load_dataset("lewtun/music_genres")
combined = concatenate_datasets([ds["train"], ds["test"]])
combined.save_to_disk("music_genres_local")
```

> **Note:** the dataset is ~10 GB. Make sure you have enough disk space before downloading. We recommend running all notebooks on **Kaggle** (free GPU, 30h/week) to avoid local storage and compute constraints.

---

## Training

All models are trained with:

- Stratified train / validation / test split
- Checkpoint saving (best model by validation accuracy)
- Early stopping to prevent overfitting
- Dropout and batch normalisation for regularisation

To train the CNN:
```bash
# Run notebook 03_cnn.ipynb on Kaggle
```

To train the LSTM:
```bash
# Run notebook 04_lstm.ipynb on Kaggle
```

---

## Evaluation

Models are evaluated using:

- Accuracy
- Weighted F1-score
- Per-class F1-score
- Confusion matrix
- ROC-AUC (one-vs-rest)

Results across all models are compared in `notebooks/04_lstm.ipynb` (evaluation section).


## Course

**ELEC3612 — Pattern Recognition and Machine Intelligence**  
School of Electrical and Information Engineering  
University of Sydney, 2025
