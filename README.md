# Version Identification - A Lyrics-Centered Approach

> Scalable Version Identification using lyrics-aligned audio embeddings.

> Our Lyrics-Informed Version Identification (LIVI) approach learns an efficient audio representation that *aligns* with a frozen, transcription-driven lyrics space—delivering competitive retrieval accuracy without running Automatic Speech Transcription (ASR) at inference.

---

## 1. Overview

**Task.** Version Identification, also known as Cover Detection, aims to recognize distinct renditions of the same underlying musical work, a task central to catalog management, copyright enforcement, and music retrieval.
State-of-the-art approaches have largely focused on harmonic and melodic features, employing increasingly complex audio pipelines designed to be invariant to musical attributes that often vary widely across covers. While effective, these methods demand substantial training time and computational resources.
By contrast, lyrics constitute a strong invariant across covers, though their use has been limited by the difficulty of extracting them accurately and efficiently from polyphonic audio.
Early methods relied on simple frameworks that limited downstream performance, while more recent systems deliver stronger results but require large models integrated within complex multimodal architectures.
We introduce LIVI, a *Lyrics-Informed Version Identification* framework that seeks to balance retrieval accuracy with computational efficiency. Leveraging supervision from state-of-the-art transcription and text embedding models, LIVI achieves retrieval accuracy on par with—or superior to—audio-based systems, while remaining lightweight and efficient by eliminating the transcription step, challenging the dominance of complexity-heavy multimodal pipelines.

**Approach.** LIVI uses a **frozen text pipeline**—ASR (Whisper) followed by a multilingual sentence encoder—to define a *lyrics-informed embedding space*. An **audio encoder** (frozen Whisper encoder + lightweight projection) is then trained to *project raw audio directly into that space*, removing ASR from inference. The training objective combines **pointwise audio-lyrics** alignment with **geometry preservation** of pairwise similarities in the target text space.

---

## 2. Repository layout

```
src/livi/
  apps/
    frozen_encoder/     # vocal detection + ASR + text embeddings (Typer CLI: livi-frozen-encoder)
    audio_encoder/      # training & inference for audio->lyrics space (CLI: livi-audio-encoder)
    retrieval_eval/     # ranking, metrics (CLI: livi-retrieval-eval)
    audio_baselines/    # audio baselines inference, checkpoints
  core/                 # I/O helpers (e.g., get_embeddings), utilities
```

Each app exposes a Typer CLI with rich `--help`.

---

## 3. Installation

### Quick start (Poetry)

System prerequisites:

- Linux (Ubuntu-like recommended), Python >= 3.11
- NVIDIA GPU + CUDA drivers for GPU runs
- `ffmpeg` for audio preprocessing

```bash
# from repo root
poetry install
poetry shell   # optional
```

### Quick start (Docker)

If you use the project Makefile:

```bash
make build
make run-bash # To open a shell
make run  # To launch the container (poetry run python3 -m livi.main is called)
```

---

## 4. Data organization

Recommended tree at repo root:

```
data/
  raw/
    audio/                  # .mp3/.wav; filenames as <version_id> or <md5_encoded>
    metadata/               # CSV(s): e.g., covers80.csv, shs100k.csv
  processed/
    vocals/                 # Results from vocal detection
    text_transcriptions/    # CSV/JSON with vocal segments, chunks, transcripts
    text_embeddings/        # .npz (id -> vector | list[vectors]) or per-chunk .npy
    audio_embeddings/       # same structure as above
results/
  metrics/
  figures/
```

**Metadata CSV (minimum columns).**

- `version_id` (identifier used everywhere)  
- `md5_encoded` (filenames identifiers)  
  
Keep identifiers consistent across CSV and filenames.

---

## 5. Model checkpoints

- **LIVI audio encoder**: download the checkpoint following this [url](https://drive.google.com/drive/folders/1hqkj7E1L2Tj-DwIbGXyRDcSGI_REq5Kf?usp=sharing) and place the `.pth` under:
  `
  src/livi/apps/audio_encoder/checkpoints/livi.pth
  `

- **Audio baselines** (optional):
  - Download the checkpoint following this [url](https://zenodo.org/records/15045900)
  - Name them: `bytecover` (we use bytecover2x), `clews`, `cqtnet`, `dvinet`
  - And put checkpoints under:
  `
  src/livi/apps/audio_baselines/checkpoints/
  `
  
---

## 6. Main pipelines

### 6.1 Prepare metadata & audio

- Place audio in `data/raw/audio/` and CSV in `data/raw/metadata/`.
- Ensure 1–1 mapping between `version_id` and  `md5_encoded` (filenames).

### 6.2 Generate lyrics embeddings (frozen encoder)

Detect vocal segments, transcribe with Whisper, then embed text (chunk-level and/or track-level).

```bash
poetry run livi-frozen-encoder inference \
  --audio-dir data/raw/audio \
  --out-path data/processed/text_embeddings/text.npz  
```

### 6.3 Generate audio embeddings (LIVI audio encoder)

Detect vocal segments, then embed audio (chunk-level).

```bash
poetry run livi-audio-encoder inference \
  --audio-dir data/raw/audio \
  --out-path data/processed/audio_embeddings/audio.npz  
```

### 6.4 Retrieval evaluation

```bash
poetry run livi-retrieval-eval evaluate   
  --path-metadata   data/raw/metadata/benchmark.csv   
  --path-embeddings data/processed/text_embeddings/text.npz   
  --col-id version_id   
  --text-id lyrics   
  --k 100   
  --path-metrics results/metrics/metrics.csv
```

## 7. Training

Default configuration files are provided:

- **Training:** [`src/livi/apps/audio_encoder/config/livi.yaml`](src/livi/apps/audio_encoder/config/livi.yaml)  
- **Inference:** [`src/livi/apps/audio_encoder/config/infer.yaml`](src/livi/apps/audio_encoder/config/infer.yaml)  

These YAMLs include the exact hyperparameters and settings reported in the paper.  

**Training.** To launch training, run the following command. The trainer uses AdamW, linear warmup, and a combined cosine + geometry-preserving loss.

```bash
poetry run livi-audio-encoder launch-training
```

**Training data recipe (high-level).** You need pairs \((z_i, t_i)\):  

- \(t_i\): lyrics embeddings from the frozen encoder (Sec. 6.2).  
- \(z_i\): precomputed Whisper log-Mel features from 30s vocal segments.  

Assemble them into WebDataset shards for scalable training. All the code to create the dataset is available in the repo under
`
src/core/data/
`
and
`
src/apps/frozen_encoder/
`.

---

## 8. Embeddings & file formats

- **NPZ layout.** `id → vector` (global) or `id → list[vectors]` (chunk-level).
- **Loader.** `src/livi/core/data/utils/io_toolbox.py::get_embeddings(path, get_single_embedding=False)`  
  - With `get_single_embedding=True`, chunk vectors are averaged per `id`.
- **Transcriptions CSV/JSON.** Fields typically include:
  - `res_detection`: list of `{start, end, vocalness}`  
  - `chunks_sec`: list of `(start, end)` used to cut audio  
  - `texts`: results from transcription (chunk-level) 
  - `join`: results from transcription (track-level)

---

## 9. Results (short summary)

- **Lyrics space alone** (frozen ASR + multilingual sentence encoder) provides a strong supervisory signal and near-ceiling accuracy when editorial lyrics are available.  
- **LIVI audio encoder** aligns audio with that space and **removes ASR at inference**, maintaining high retrieval performance while cutting latency (~3–6× faster inference vs. strong audio baselines; ~20× vs. Whisper decoding).  
- On standard benchmarks (Covers80, SHS100k) and large-scale Discogs-VI, LIVI is competitive with or superior to state-of-the-art audio-only systems, especially in HR@1 and mAP@10, with a compact model (≈32M params).

*(See the paper for full tables, runtime profiles, and ablations.)*

## 12. Citation

If you use LIVI or this codebase in academic work, please cite the associated paper:

> **Scalable Music Cover Retrieval Using Lyrics-Aligned Audio Embeddings** — J. Affolter, B. Martin, E. V. Epure, G. Meseguer-Brocal, F. Kaplan.

---

**Contact.** Joanne Affolter (EPFL — Deezer). Contributions and issues are welcome.
