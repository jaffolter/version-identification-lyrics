import re
from typing import List, Optional, Union, Tuple, Dict, Set
import ast

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from functools import lru_cache
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
from loguru import logger

from collections import defaultdict

from livi.core.data.utils.audio_toolbox import load_audio, split_audio_30s, split_audio_predefined


class Transcriber:
    """
    Wrapper around a Hugging Face Whisper model for transcribing
    a list of audio chunks into cleaned text.

    Workflow
    --------
    1. Load model and processor with chosen dtype/device.
    2. Batch audio chunks → run `.generate()` → decode to strings.
    3. Post-process:
    - remove unwanted phrases,
    - collapse repeated words,
    - drop very short/empty outputs.
    4. Join remaining segments with double newlines.

    Parameters
    ----------
    model_name : str
        Hugging Face model ID (e.g., "openai/whisper-large-v3-turbo").
    device : str, optional
        Device to run inference on ("cuda" or "cpu"). Default: auto-detect ("cuda" if available).
    dtype_fp16_on_cuda : bool, default=True
        Use float16 precision on CUDA if available, else float32.
    sampling_rate : int, default=16000
        Expected input audio sampling rate (Hz).

    Generation parameters
    ---------------------
    num_beams : int
    condition_on_prev_tokens : bool
    compression_ratio_threshold : float
    temperature : tuple[float, ...]
    logprob_threshold : float
    return_timestamps : bool

    Cleaning parameters
    -------------------
    remove_phrases : list[str]
        Exact phrases to remove (case-insensitive, whole words).
    repeat_threshold : int
        Collapse words repeated at least this many times.
    min_words_per_chunk : int
        Discard chunk-level text shorter than this.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        *,
        device: Optional[str] = None,
        dtype_fp16_on_cuda: bool = True,
        sampling_rate: int = 16000,
        num_beams: int = 1,
        condition_on_prev_tokens: bool = False,
        compression_ratio_threshold: float = 1.35,
        temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        logprob_threshold: float = -1.0,
        return_timestamps: bool = True,
        remove_phrases: Optional[List[str]] = None,
        repeat_threshold: int = 3,
        min_words_per_chunk: int = 4,
    ) -> None:
        # -------- device / dtype ----------
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = dtype_fp16_on_cuda and torch.cuda.is_available()
        self.torch_dtype = torch.float16 if use_fp16 else torch.float32
        self.sampling_rate = int(sampling_rate)

        # -------- model / processor -------
        self.model: AutoModelForSpeechSeq2Seq = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(model_name)

        # -------- generation args --------
        self.num_beams = int(num_beams)
        self.condition_on_prev_tokens = bool(condition_on_prev_tokens)
        self.compression_ratio_threshold = float(compression_ratio_threshold)
        self.temperature = tuple(temperature)
        self.logprob_threshold = float(logprob_threshold)
        self.return_timestamps = bool(return_timestamps)

        # -------- cleaning args ----------
        self.remove_phrases = remove_phrases if remove_phrases is not None else ["Thank you.", "music"]
        self.repeat_threshold = int(repeat_threshold)
        self.min_words_per_chunk = int(min_words_per_chunk)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _build_gen_kwargs(self, translate: bool) -> dict:
        """
        Build generation kwargs from the object attributes.
        If `translate` is True, add `language="en"` to the kwargs.
        """
        gen_kwargs = {
            "num_beams": self.num_beams,
            "condition_on_prev_tokens": self.condition_on_prev_tokens,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "temperature": self.temperature,
            "logprob_threshold": self.logprob_threshold,
            "return_timestamps": self.return_timestamps,
        }
        if translate:
            gen_kwargs["language"] = "en"
        return gen_kwargs

    def transcribe(
        self,
        audio_chunks: List[np.ndarray],
        translate: bool = False,
    ) -> Optional[Tuple[List[str], List[bool], str]]:
        """
        Transcribe a list of audio chunks, clean + filter results, and
        return text(s) with a boolean mask indicating which chunks succeeded.

        Parameters
        ----------
        audio_chunks : list
            List of 1D arrays at `self.sampling_rate`.
        translate : bool
            If True, set gen_kwargs["language"] = "en".

        Returns
        -------
        Optional[Tuple[List[str], List[bool], str]]
            - (texts, mask), where texts is the list of
            per-chunk strings, mask is a parallel list of booleans
        """
        # ---- feature extraction with fallback ----
        processed = self.processor(
            audio_chunks,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            return_attention_mask=True,
        )
        if processed["input_features"].shape[-1] < 3000:
            processed = self.processor(
                audio_chunks,
                return_tensors="pt",
                sampling_rate=self.sampling_rate,
            )

        processed = {k: v.to(self.device, dtype=self.torch_dtype) for k, v in processed.items()}

        # ---- gen kwargs ----
        gen_kwargs = self._build_gen_kwargs(translate)

        # ---- forward pass ----
        with torch.no_grad():
            pred_ids = self.model.generate(**processed, **gen_kwargs)
            raw_texts = self.processor.batch_decode(pred_ids, skip_special_tokens=True)

        # ---- cleaning ----
        cleaned = self.clean_transcription(raw_texts)

        # ---- filtering: keep if not empty & has enough words ----
        mask = [t.strip() != "" and len(t.split()) >= self.min_words_per_chunk for t in cleaned]
        texts = [t if ok else "" for t, ok in zip(cleaned, mask)]

        # joined_text = "\n\n".join([t for t, ok in zip(texts, mask) if ok])
        return texts, mask

    # ---------------------------------------------------------------------
    # Cleaning utilities
    # ---------------------------------------------------------------------
    def clean_transcription(self, pred_text: List[str]) -> List[str]:
        """
        Apply cleanup:
        1) Remove phrases in `self.remove_phrases` (whole-word, case-insensitive).
        2) Collapse repeated words past `self.repeat_threshold`.
        """

        def clean_text(text: str) -> str:
            out = text
            for phrase in self.remove_phrases:
                out = re.sub(rf"\b{re.escape(phrase)}\b", "", out, flags=re.IGNORECASE)
            return out.strip()

        def collapse_repeated_words(text: str) -> str:
            if self.repeat_threshold <= 1:
                return text
            pattern = re.compile(
                rf"\b(\w+)([,\s]+(?:\1\b[,\s]+){{{self.repeat_threshold - 1},}})",
                flags=re.IGNORECASE,
            )
            return pattern.sub(r"\1 ", text)

        return [collapse_repeated_words(clean_text(x)) for x in pred_text]


# --------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------


@lru_cache(maxsize=2)
def _get_cached_transcriber(
    model_name: str = "openai/whisper-large-v3-turbo",
    device: Optional[str] = None,
    dtype_fp16_on_cuda: bool = True,
    sampling_rate: int = 16000,
    num_beams: int = 1,
    condition_on_prev_tokens: bool = False,
    compression_ratio_threshold: float = 1.35,
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    logprob_threshold: float = -1.0,
    return_timestamps: bool = True,
    remove_phrases: Optional[tuple[str, ...]] = ("Thank you.", "music"),
    repeat_threshold: int = 3,
    min_words_per_chunk: int = 4,
) -> Transcriber:
    """
    Build (or reuse cached) Transcriber. LRU cache ensures model is
    loaded only once per unique parameter configuration.
    """
    return Transcriber(
        model_name=model_name,
        device=device,
        dtype_fp16_on_cuda=dtype_fp16_on_cuda,
        sampling_rate=sampling_rate,
        num_beams=num_beams,
        condition_on_prev_tokens=condition_on_prev_tokens,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=temperature,
        logprob_threshold=logprob_threshold,
        return_timestamps=return_timestamps,
        remove_phrases=list(remove_phrases) if remove_phrases else None,
        repeat_threshold=repeat_threshold,
        min_words_per_chunk=min_words_per_chunk,
    )


def transcribe(
    audio_chunks: List[np.ndarray],
    translate: bool = False,
    *,
    transcriber: Optional[Transcriber] = None,
    model_name: str = "openai/whisper-large-v3-turbo",
    device: Optional[str] = None,
    dtype_fp16_on_cuda: bool = True,
    sampling_rate: int = 16000,
    num_beams: int = 1,
    condition_on_prev_tokens: bool = False,
    compression_ratio_threshold: float = 1.35,
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    logprob_threshold: float = -1.0,
    return_timestamps: bool = True,
    remove_phrases: Optional[tuple[str, ...]] = ("Thank you.", "music"),
    repeat_threshold: int = 3,
    min_words_per_chunk: int = 4,
) -> Optional[Tuple[List[str], List[bool], str]]:
    """
    High-level helper to run transcription without manually
    instantiating a Transcriber. Uses cached models to avoid reload.

    Parameters
    ----------
    audio_chunks : list
        List of 1D arrays/tensors at 16 kHz.
    translate : bool
        Force English output if True.
    All other params are passed to the underlying Transcriber.

    Returns
    -------
    Optional[Tuple[List[str], List[bool], str]]
        - (texts, mask, joined_text)
    """
    transcriber = transcriber or _get_cached_transcriber(
        model_name=model_name,
        device=device,
        dtype_fp16_on_cuda=dtype_fp16_on_cuda,
        sampling_rate=sampling_rate,
        num_beams=num_beams,
        condition_on_prev_tokens=condition_on_prev_tokens,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=temperature,
        logprob_threshold=logprob_threshold,
        return_timestamps=return_timestamps,
        remove_phrases=remove_phrases,
        repeat_threshold=repeat_threshold,
        min_words_per_chunk=min_words_per_chunk,
    )

    return transcriber.transcribe(audio_chunks, translate=translate)


def transcribe_batch(
    audio_chunks_by_track: Dict[str, List[Union[list, np.ndarray, torch.Tensor]]],
    translate: bool = False,
    batch_size: int = 8,
    *,
    transcriber: Optional[Transcriber] = None,
    model_name: str = "openai/whisper-large-v3-turbo",
    device: Optional[str] = None,
    dtype_fp16_on_cuda: bool = True,
    sampling_rate: int = 16000,
    num_beams: int = 1,
    condition_on_prev_tokens: bool = False,
    compression_ratio_threshold: float = 1.35,
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    logprob_threshold: float = -1.0,
    return_timestamps: bool = True,
    remove_phrases: Optional[tuple[str, ...]] = ("Thank you.", "music"),
    repeat_threshold: int = 3,
    min_words_per_chunk: int = 4,
) -> Optional[Tuple[List[str], List[bool], str]]:
    """
    High-level helper to run transcription on multiple tracks in a single call.
    Flatten all chunks across tracks -> run batched transcription -> regroup by track.

    Parameters
    ----------
    audio_chunks : list
        List of 1D arrays/tensors at 16 kHz.
    translate : bool
        Force English output if True.
    batch_size: int
        Number of audio chunks to process in parallel.
    All other params are passed to the underlying Transcriber.

    Returns
    -------
    Optional[Tuple[List[str], List[bool], str]]
        - (joined_text, mask, joined_text)
    """
    transcriber = transcriber or _get_cached_transcriber(
        model_name=model_name,
        device=device,
        dtype_fp16_on_cuda=dtype_fp16_on_cuda,
        sampling_rate=sampling_rate,
        num_beams=num_beams,
        condition_on_prev_tokens=condition_on_prev_tokens,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=temperature,
        logprob_threshold=logprob_threshold,
        return_timestamps=return_timestamps,
        remove_phrases=remove_phrases,
        repeat_threshold=repeat_threshold,
        min_words_per_chunk=min_words_per_chunk,
    )
    # ---------- flatten ----------
    all_chunks: List[np.ndarray] = []
    chunk_sources: List[str] = []
    for track_id, chunks in audio_chunks_by_track.items():
        all_chunks.extend(chunks)
        chunk_sources.extend([track_id] * len(chunks))

    # ---------- batched inference (keep alignment) ----------
    flat_texts: List[str] = []
    flat_mask: List[bool] = []

    for i in tqdm(
        range(0, len(all_chunks), batch_size),
        total=(len(all_chunks) + batch_size - 1) // batch_size,
        desc="Transcribing batches",
    ):
        batch_chunks = all_chunks[i : i + batch_size]

        # ask for per-chunk outputs so we keep 1:1 mapping
        out = transcriber.transcribe(batch_chunks, translate=translate)

        if out is None:
            # none of the chunks in this batch produced valid text;
            # we still need placeholders to keep alignment
            flat_texts.extend([""] * len(batch_chunks))
            flat_mask.extend([False] * len(batch_chunks))
        else:
            texts, mask = out
            flat_texts.extend(texts)
            flat_mask.extend(mask)

    # sanity: lengths must match
    assert len(flat_texts) == len(all_chunks) == len(flat_mask)

    # ---------- regroup to per-track ----------
    per_track_texts: Dict[str, List[str]] = defaultdict(list)
    per_track_mask: Dict[str, List[bool]] = defaultdict(list)

    for src, txt, ok in zip(chunk_sources, flat_texts, flat_mask):
        per_track_texts[src].append(txt)
        per_track_mask[src].append(ok)

    # joined strings per track (only valid chunks)
    results: Dict[str, Dict[str, Union[str, List[str], List[bool]]]] = {}
    for track_id in audio_chunks_by_track.keys():
        texts = per_track_texts.get(track_id, [])
        mask = per_track_mask.get(track_id, [])
        joined = "\n\n".join([t for t, ok in zip(texts, mask) if ok]) if texts else ""
        results[track_id] = {"texts": texts, "mask": mask, "joined": joined}

    return results


def transcribe_dataset(
    path_metadata: Path,
    dir_audio: Path,
    path_output: Path,
    vocal: Optional[bool] = True,
    translate: Optional[bool] = False,
    batch_size: Optional[int] = 64,
    nb_tracks_max: Optional[int] = 32,
    chunk_duration: Optional[float] = 30.0,
    *,
    transcriber: Optional[Transcriber] = None,
    model_name: str = "openai/whisper-large-v3-turbo",
    device: Optional[str] = None,
    dtype_fp16_on_cuda: bool = True,
    sampling_rate: int = 16000,
    num_beams: int = 1,
    condition_on_prev_tokens: bool = False,
    compression_ratio_threshold: float = 1.35,
    temperature: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    logprob_threshold: float = -1.0,
    return_timestamps: bool = True,
    remove_phrases: Optional[tuple[str, ...]] = ("Thank you.", "music"),
    repeat_threshold: int = 3,
    min_words_per_chunk: int = 4,
):
    """
    Main function to transcribe audio from all tracks in a dataset.

    If vocal=True, a call to vocal chunks extraction should be done first.

    Args:
        path_metadata (Path): path to csv file containing metadata
        dir_audio (Path): path to directory containing audio files
        path_output (Path): path to output csv file to store transcriptions
        vocal (Optional[bool]): whether vocal detection was performed before,
            and transcription should be performed only on vocal segments
        translate (Optional[bool]): whether to translate transcriptions
        batch_size (Optional[int]): batch size for processing
        nb_tracks_max (Optional[int]): max number of tracks to accumulate before feeding to transcription
        chunk_duration (Optional[float]): duration of audio chunks in seconds
    """
    transcriber = transcriber or _get_cached_transcriber(
        model_name=model_name,
        device=device,
        dtype_fp16_on_cuda=dtype_fp16_on_cuda,
        sampling_rate=sampling_rate,
        num_beams=num_beams,
        condition_on_prev_tokens=condition_on_prev_tokens,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=temperature,
        logprob_threshold=logprob_threshold,
        return_timestamps=return_timestamps,
        remove_phrases=remove_phrases,
        repeat_threshold=repeat_threshold,
        min_words_per_chunk=min_words_per_chunk,
    )

    if not os.path.exists(path_metadata):
        raise ValueError(f"Metadata path {path_metadata} does not exist.")

    if not os.path.exists(dir_audio):
        raise ValueError(f"Audio directory {dir_audio} does not exist.")

    if not os.path.exists(path_output):
        os.makedirs(os.path.dirname(path_output), exist_ok=True)

    # Read metadata about tracks to transcribe
    df = pd.read_csv(path_metadata)

    # Variables to store results
    processed_ids: Set[str] = set()
    results_df: pd.DataFrame = pd.DataFrame(columns=["md5_encoded", "texts", "mask", "joined"])
    audio_chunks_by_track: Dict[str, List[Union[list, np.ndarray, torch.Tensor]]] = {}

    # Check if output CSV already exists and filter out already processed files
    if os.path.exists(path_output):
        existing_df = pd.read_csv(path_output)
        processed_ids = set(existing_df["md5_encoded"])
        logger.info(f"Found existing output: {len(processed_ids)}/{len(df)} transcriptions already done.")

        # Filter out already processed
        df = df[~df["md5_encoded"].isin(processed_ids)]

        # If no files left to process, exit
        if df.empty:
            logger.success("✅ All files already transcribed!")
            exit()

        # Define already processed transcriptions
        results_df = existing_df

    # Start transcription in batches
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Transcribing")):
        try:
            track_id = row["md5_encoded"]
            version_id = row.get("version_id", None)

            # Load audio file
            audio_path = dir_audio / f"{track_id}.mp3"

            # Data Preprocessing ---------
            # If using vocal segments as inputs, retrieve the (start, end) from metadata
            if vocal:
                vocal_segments_metadata = ast.literal_eval(row["chunks_sec"])
                audio_chunks = split_audio_predefined(audio_path, vocal_segments_metadata)

            # Otherwise, split the audio into 30s chunks
            else:
                audio_chunks = split_audio_30s(
                    audio_path=audio_path, sample_rate=sampling_rate, chunk_duration=chunk_duration
                )

            # If we have audio chunks, then append to the the dict for transcription
            if len(audio_chunks) > 0:
                audio_chunks_by_track[track_id] = audio_chunks

            # We can transcribe the accumulated tracks
            if len(audio_chunks_by_track) >= nb_tracks_max:
                transcriptions_results = transcribe_batch(
                    audio_chunks_by_track, translate=translate, batch_size=batch_size
                )

                # Retrieve results and update transcriptions DataFrame
                batch_rows = [
                    {
                        "version_id": version_id,
                        "md5_encoded": id,
                        "texts": res["texts"],
                        "mask": res["mask"],
                        "joined": res["joined"],
                    }
                    for id, res in transcriptions_results.items()
                ]
                batch_df = pd.DataFrame(batch_rows)
                results_df = pd.concat([results_df, batch_df], ignore_index=True)

                # Save updated file
                results_df.to_csv(path_output, index=False)
                logger.info(f"💾 Updated saved to {path_output}")

                audio_chunks_by_track = {}

        except Exception as e:
            logger.error(f"Error processing {track_id}: {e}")
            continue

    # ---- Final batch ----
    if audio_chunks_by_track:
        logger.info(f"Transcribing final batch ({len(audio_chunks_by_track)} songs)")
        transcriptions_results = transcribe_batch(audio_chunks_by_track, translate=translate, batch_size=batch_size)

        # Retrieve results and update transcriptions DataFrame
        batch_rows = [
            {
                "version_id": version_id,
                "md5_encoded": id,
                "texts": res["texts"],
                "mask": res["mask"],
                "joined": res["joined"],
            }
            for id, res in transcriptions_results.items()
        ]
        batch_df = pd.DataFrame(batch_rows)
        results_df = pd.concat([results_df, batch_df], ignore_index=True)
        results_df.to_csv(path_output, index=False)
