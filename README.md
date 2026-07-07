# Lightweight Swin Transformer for Speech Emotion Recognition

[![Paper](https://img.shields.io/badge/Paper-Accepted-green.svg)](#) <!-- Add your paper link here later -->
[![Parameters](https://img.shields.io/badge/Parameters-0.85M-blue.svg)](#)
[![GFLOPs](https://img.shields.io/badge/GFLOPs-0.32-orange.svg)](#)

This is the official PyTorch implementation of the paper:
**"Lightweight Swin Transformer with Prosodic Tokens and Huber-Based HuBERT Distillation for Cross-Lingual Speech Emotion Recognition"** *(Accepted in International Journal of Intelligent Engineering & Systems - IJIES)*.

## 📌 Overview
This repository provides the code and data splitting protocols to reproduce the results of our highly compact and accurate Speech Emotion Recognition (SER) framework. The model achieves State-of-the-Art performance while requiring only **0.85M parameters**.

### Key Contributions:
1. **Dual-Branch Input:** Fuses high-resolution log-Mel spectrograms with lightweight explicit prosodic tokens (pitch, energy, ZCR, rate).
2. **Progressive Receptive Field:** Stage-wise local-window attention schedule (3×3 → 5×5 → 7×7 → 7×7) to aggregate local transients and long-range prosody dynamically.
3. **Huber-Based Knowledge Distillation:** Uses a frozen HuBERT teacher (16kHz raw waveform input) to distill acoustic priors into the student (2D spectrogram input) via a robust Huber alignment loss.

---

## 📂 Repository Structure
├── data_splits/          # Strictly speaker-independent train/test split files (.txt) for 5 datasets
├── preprocess.py         # Scripts for log-Mel extraction and explicit prosodic tokens (F0, Energy, ZCR)
├── model.py              # The lightweight Swin Transformer architecture & prosodic token injection
├── train.py              # Training loop including Huber-guided knowledge distillation
├── evaluate.py           # Evaluation script for within-corpus and zero-shot cross-lingual tasks
└── requirements.txt      # Python dependencies


## 🛠️ Requirements
- Python 3.8+
- PyTorch >= 1.10.0
- Torchaudio
- Librosa
- Transformers (HuggingFace, for frozen HuBERT teacher)

Install dependencies:
```bash
pip install -r requirements.txt

## 🚀 Reproducibility
To address data leakage concerns and ensure 100% reproducibility, all data augmentations (SpecAugment, time-stretching, pitch-shifting, noise injection) are applied dynamically on-the-fly AFTER the speaker-independent dataset splitting.

Check the data_splits/ directory for the exact .txt files used to partition CREMA-D, EMO-DB, RAVDESS, SAVEE, and TESS.

Extract Features:
python preprocess.py --dataset_dir /path/to/dataset --out_dir /path/to/save/features
Training:
python train.py --config configs/default.yaml --distillation True --teacher_model "hubert-base-ls960"

## 📊 Results SummaryOur compact student (0.85M parameters) outperforms massive self-supervised models and prior hybrid networks:
Dataset,Accuracy (%),Macro F1
CREMA-D,94.02 ± 0.19,0.94
EMO-DB,96.32 ± 0.15,0.96
RAVDESS,96.59 ± 0.17,0.96
SAVEE,98.08 ± 0.22,0.98
TESS,99.89 ± 0.05,1.00

Zero-shot Cross-lingual (English → German): 63.44% Weighted Accuracy.

## 📖 Citation
If you find this code or our paper useful for your research, please cite:
@article{alkhafaji202xlightweight,
  title={Lightweight Swin Transformer with Prosodic Tokens and Huber-Based HuBERT Distillation for Cross-Lingual Speech Emotion Recognition},
  author={Alkhafaji, Maather and Lakizadeh, Amir},
  journal={International Journal of Intelligent Engineering & Systems},
  year={202x}
}
