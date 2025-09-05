"""
Discogs-VI: build training dataset from transcriptions + vocal detector results.
(Needed to re-align the number of audio chunks with the number of transcribed chunks, which was
not saved during transcription) 
How: same preprocessing on vocal chunks outputs from the vocal detector as before feeding to Whisper, 
then if nb_chunks < nb_transcriptions: we remove the vocal chunks with the smallest duration

---------------------------------------------------------------------------
Purpose
    Merge a metadata table with:
      - transcriptions (transcriptions.csv)
      - vocal detector results (vocal_detector_results.csv)
    Then, generate padded/merged ≤30s chunks for vocal regions and align them
    with transcription segments.

Inputs
    1) Base metadata CSV with at least:
         version_id, clique_id, md5_encoded
       (optional but supported if present: deezer_id or deezer_track_id)
    2) Merged transcriptions CSV:
         must contain either 'transcription_vocal' or 'transcription',
         and a 'file' ('<md5>.<ext>') or a 'md5_encoded' column.
    3) Merged vocal detector CSV:
         must contain 'vocal_segments' (stringified list of {"start","end"}),
         and a 'file' or 'md5_encoded' column.
         Optional: 'vocalness_score' or 'vocalness'.

Outputs
    - A single CSV with one row per chunk:
        version_id, clique_id, md5_encoded,
        (optional) deezer_id, vocalness,
        chunk_id, start, end, duration, transcription

Configuration
    - You can pass paths via CLI options.
    - You can also surface defaults via your project config (e.g., livi.yaml).
---------------------------------------------------------------------------

Typical usage (CLI)
    poetry run livi-data build-discogs-vi-chunks \
        --base-metadata data/metadata_base.csv \
        --transcriptions data/transcriptions.csv \
        --vocals data/vocal_detector_results.csv \
        --output data/metadata_chunks.csv
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd


# ---------------------------------------------------------------------
# CSV loaders + normalizers
# ---------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def normalize_transcriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure transcriptions df has:
      - 'md5_encoded' (derived from 'file' if needed)
      - 'transcription_vocal' (renamed from 'transcription' if needed)
    """
    cols = set(df.columns)

    if "md5_encoded" not in cols:
        if "file" not in cols:
            raise ValueError("Transcriptions CSV must have 'md5_encoded' or 'file'.")
        df["md5_encoded"] = df["file"].apply(lambda x: str(x).split(".")[0])

    if "transcription_vocal" not in cols:
        if "transcription" in cols:
            df = df.rename(columns={"transcription": "transcription_vocal"})
        else:
            raise ValueError("Transcriptions CSV must have 'transcription_vocal' or 'transcription'.")

    return df[["md5_encoded", "transcription_vocal"]].copy()


def normalize_vocals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure vocals df has:
      - 'md5_encoded' (derived from 'file' if needed)
      - 'vocal_segments'
      - optional 'vocalness' (renamed from 'vocalness_score' if present)
    """
    cols = set(df.columns)

    if "md5_encoded" not in cols:
        if "file" not in cols:
            raise ValueError("Vocal results CSV must have 'md5_encoded' or 'file'.")
        df["md5_encoded"] = df["file"].apply(lambda x: str(x).split(".")[0])

    if "vocal_segments" not in cols:
        raise ValueError("Vocal results CSV must have 'vocal_segments' column.")

    # normalize vocalness column name if present
    if "vocalness" in cols:
        keep = ["md5_encoded", "vocal_segments", "vocalness"]
    elif "vocalness_score" in cols:
        df = df.rename(columns={"vocalness_score": "vocalness"})
        keep = ["md5_encoded", "vocal_segments", "vocalness"]
    else:
        keep = ["md5_encoded", "vocal_segments"]

    return df[keep].copy()


def merge_all(
    base_df: pd.DataFrame,
    trans_df: pd.DataFrame,
    vocal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join base metadata with transcriptions and vocal segments on md5_encoded.
    Keeps only rows with both transcription_vocal and vocal_segments.
    Drops duplicate version_id if present.
    """
    merged = (
        base_df.merge(trans_df, on="md5_encoded", how="left")
               .merge(vocal_df, on="md5_encoded", how="left")
    )
    # Drop duplicates on version_id if that column exists
    if "version_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["version_id"])

    # Keep rows with both fields present
    merged = merged[
        merged["transcription_vocal"].notna() &
        merged["vocal_segments"].notna()
    ].copy()

    return merged


# ---------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------
def get_audio_chunks(
    vocal_segments: List[dict],
    max_chunk_sec: float = 30.0,
    max_pad_sec: float = 10.0,
) -> List[Tuple[float, float, float]]:
    """
    1) Pad each vocal segment up to 30s with at most 10s total padding (split on both sides).
    2) Merge overlaps.
    3) Slice into consecutive chunks of ≤ max_chunk_sec.
    Returns list of (start, end, duration).
    """
    padded: List[Tuple[float, float]] = []
    for seg in vocal_segments:
        s, e = float(seg["start"]), float(seg["end"])
        dur = e - s
        if dur >= max_chunk_sec:
            padded.append((s, e))
            continue
        pad_each_side = min(max_pad_sec / 2.0, (max_chunk_sec - dur) / 2.0)
        padded.append((max(0.0, s - pad_each_side), e + pad_each_side))

    padded.sort()
    merged: List[List[float]] = []
    for s, e in padded:
        if not merged or merged[-1][1] < s:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    chunks: List[Tuple[float, float, float]] = []
    for s, e in merged:
        cur = s
        while e - cur > max_chunk_sec:
            chunks.append((cur, cur + max_chunk_sec, max_chunk_sec))
            cur += max_chunk_sec
        chunks.append((cur, e, e - cur))
    return chunks


def build_chunks_rows(
    df: pd.DataFrame,
    transcription_split_sep: str = "\n\n",
    max_chunk_sec: float = 30.0,
) -> pd.DataFrame:
    """
    For each row:
      - parse vocal_segments
      - create ≤30s chunks via get_audio_chunks
      - split transcription_vocal by `transcription_split_sep`
      - align counts (keep longest chunks if more chunks than segments; truncate segments otherwise)

    Returns one row per chunk with metadata + chunk timing + transcription slice.
    """
    out = []

    # pick optional columns safely
    has_deezer_track_id = "deezer_track_id" in df.columns
    has_deezer_id = "deezer_id" in df.columns
    has_vocalness = "vocalness" in df.columns

    for row in df.itertuples(index=False):
        # transcription segments
        t_full = str(getattr(row, "transcription_vocal", ""))
        t_segments = [seg for seg in t_full.split(transcription_split_sep) if seg.strip()]

        # vocal segments
        segments = ast.literal_eval(getattr(row, "vocal_segments"))
        chunks = get_audio_chunks(segments, max_chunk_sec=max_chunk_sec)

        nb_chunks, nb_t = len(chunks), len(t_segments)
        if nb_chunks > nb_t:
            chunks = sorted(chunks, key=lambda x: x[2], reverse=True)[:nb_t]
            chunks = sorted(chunks, key=lambda x: x[0])
        elif nb_t > nb_chunks:
            t_segments = t_segments[:nb_chunks]

        deezer_value: Optional[str] = None
        if has_deezer_track_id:
            deezer_value = getattr(row, "deezer_track_id")
        elif has_deezer_id:
            deezer_value = getattr(row, "deezer_id")

        for i, (start, end, dur) in enumerate(chunks):
            out.append({
                "version_id": getattr(row, "version_id") if "version_id" in df.columns else None,
                "clique_id": getattr(row, "clique_id") if "clique_id" in df.columns else None,
                "md5_encoded": getattr(row, "md5_encoded"),
                "deezer_id": deezer_value,
                "chunk_id": i,
                "start": start,
                "end": end,
                "duration": dur,
                "vocalness": getattr(row, "vocalness") if has_vocalness else None,
                "transcription": t_segments[i] if i < len(t_segments) else "",
            })

    return pd.DataFrame(out)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
def run_build_discogs_vi_chunks_merged(
    base_metadata: Path,
    transcriptions_csv: Path,
    vocals_csv: Path,
    output_csv: Path,
    transcription_split_sep: str = "\n\n",
    max_chunk_sec: float = 30.0,
) -> None:
    """
    Build chunk-level metadata from merged transcriptions + merged vocal results.
    """
    base_df = _read_csv(base_metadata)

    trans_df_raw = _read_csv(transcriptions_csv)
    trans_df = normalize_transcriptions(trans_df_raw)

    vocal_df_raw = _read_csv(vocals_csv)
    vocal_df = normalize_vocals(vocal_df_raw)

    merged = merge_all(base_df, trans_df, vocal_df)

    # optional: quick schema check
    required = {"md5_encoded", "transcription_vocal", "vocal_segments"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Missing required columns in merged dataframe: {missing}")

    chunks_df = build_chunks_rows(
        merged,
        transcription_split_sep=transcription_split_sep,
        max_chunk_sec=max_chunk_sec,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(output_csv, index=False)



if __name__ == "__main__":
    run_build_discogs_vi_chunks_merged(
        base_metadata=Path("/nfs/interns/jaffolter/data/audio_encoder_dataset/metadata/full_metadata.csv"),
        transcriptions_csv=Path("/nfs/interns/jaffolter/data/audio_encoder_dataset/metadata/transcriptions.csv"),
        vocals_csv=Path("/nfs/interns/jaffolter/data/audio_encoder_dataset/metadata/vocal_detector_results.csv"),
        output_csv=Path("/nfs/interns/jaffolter/data/audio_encoder_dataset/metadata/full2.csv"),
        transcription_split_sep="\n\n",
        max_chunk_sec=30.0,
    )