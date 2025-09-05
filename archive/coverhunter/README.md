# CoverHunter

CoverHunter is a cover song / musical **version identification** model.  
This repo adapts the authors’ implementation to serve as an **audio baseline** alongside our lyrics-based models (CQT → embeddings → retrieval/eval).

**Model weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Qw1kQw1kQw1kQw1kQw1kQw1kQw1kQw1k?usp=sharing) and should be placed in `model/checkpoints/`.**

- 📄 Paper: **“CoverHunter: Cover Song Identification with Refined Attention and Alignments”** (ICME 2023) — [arXiv:2306.09025](https://arxiv.org/abs/2306.09025)  
- 💻 Original code: [github.com/Liu-Feng-deeplearning/CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter)

---

## Pipeline Overview

1) **Dataset metadata (CSV)** in `benchmarks/<dataset>.csv`  
   Must contain at least:
   - `version_id`
   - `md5_encoded` (or adjust the path mapping in the script)

2) **CQT extraction** → writes one file per track:  `<OUTPUT_DIR>/cqt_feat/<version_id>.cqt.npy`

3) **Embedding generation** with the trained CoverHunter model:  `<OUTPUT_DIR>/<dataset_name>_embeddings.npz`

---

## Directory Layout

```
coverhunter/
├── dataset.py          # CQT dataset + collate
├── extract_cqt.py      # Audio → CQT (.npy)
├── inference.py        # Model load + embeddings
scripts/
└── run_pipeline.py     # End-to-end driver (CSV → CQT → embeddings)
model/
├── config/hparams.yaml # Hyperparameters
└── checkpoints/        # Weights
benchmarks/
└── .csv       # version_id, md5_encoded, …
```

---

## Quickstart

```bash
# Example: covers80
DATASET=covers80 poetry run python3 -m main
```

The script will:

- read benchmarks/$DATASET.csv
- extract CQTs to <OUTPUT_DIR>/cqt_feat/
- save embeddings to <OUTPUT_DIR>/${DATASET}_embeddings.npz

Set output/audio/model paths inside scripts/run_pipeline.py (look for the “EDIT THESE TO MATCH YOUR ENV” comments).

---

## Citation

If you use this baseline, please cite the original authors:

CoverHunter: Cover Song Identification with Refined Attention and Alignments -
Feng Liu, Deyi Tuo, Yinan Xu, Xintong Han. ICME 2023. -
arXiv: 2306.09025

Original repository: <https://github.com/Liu-Feng-deeplearning/CoverHunter>
