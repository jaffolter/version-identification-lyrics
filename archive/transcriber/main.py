import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from tqdm import tqdm
import pandas as pd
import os
import time
import torchaudio
import numpy as np
import re
from collections import defaultdict
import logging


class Transcriber:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.pipe = None
        self.dataset_name = dataset_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.init_model(model_name)

    def init_model(self, model_name):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)
        self.model = model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            batch_size=32,
            chunk_length_s=30,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def split_vocal_segments(
        self,
        audio_path: str,
        vocal_segments: list,
        sample_rate: int = 16_000,
        no_processing: bool = False,
    ) -> list:
        """
        Load audio and return a list of separate vocal chunks (one tensor per segment).

        Args:
            audio_path (str): Path to the audio file.
            vocal_segments (list): List of dicts with 'start' and 'end' times in seconds.
            sample_rate (int, optional): Resample to this rate.

        Returns:
            list of torch.Tensor: One waveform tensor per vocal segment.
        """
        waveform, orig_sr = torchaudio.load(audio_path)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate and sample_rate != orig_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)
            sr = sample_rate
        else:
            sr = orig_sr

        padded_segments = []

        for seg in vocal_segments:
            duration = seg["end"] - seg["start"]

            if duration > 30:
                padded_segments.append((seg["start"], seg["end"]))  # No padding
                continue

            pad = min(10, (30 - duration) / 2)
            start = max(0, seg["start"] - pad)
            end = seg["end"] + pad
            padded_segments.append((start, end))

        padded_segments.sort()
        merged_segments = []
        for start, end in padded_segments:
            if not merged_segments or merged_segments[-1][1] < start:
                merged_segments.append([start, end])
            else:
                merged_segments[-1][1] = max(merged_segments[-1][1], end)

        # Slice waveform from merged segments, in chunks of max 30s
        chunks = []
        for start, end in merged_segments:
            seg_dur = end - start
            while seg_dur > 30:
                chunk = waveform[:, int(start * sr) : int((start + 30) * sr)]
                chunks.append(chunk.squeeze(0).numpy())
                start += 30
                seg_dur = end - start
            chunk = waveform[:, int(start * sr) : int(end * sr)]
            chunks.append(chunk.squeeze(0).numpy())

        return chunks

    def split_audio(
        self, audio_path: str, sample_rate: int = 16000, chunk_duration: int = 30
    ) -> list:
        """
        Load audio and split it into fixed-length chunks.

        Args:
            audio_path (str): Path to the audio file.
            sample_rate (int): Desired sampling rate.
            chunk_duration (int): Duration of each chunk in seconds.

        Returns:
            list of torch.Tensor: List of waveform chunks.
        """
        waveform, orig_sr = torchaudio.load(audio_path)

        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)

        total_samples = waveform.shape[1]
        chunk_size = chunk_duration * sample_rate

        chunks = []
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            chunk = waveform[:, start:end]

            # pad the last chunk if it's too short
            if chunk.shape[1] < chunk_size:
                padding = torch.zeros((waveform.shape[0], chunk_size - chunk.shape[1]))
                chunk = torch.cat([chunk, padding], dim=1)

            chunks.append(chunk.squeeze(0).numpy())

        return chunks

    def transcribe_chunks(self, audio_chunks: list, translate=False) -> list:
        """
        Transcribe a list of audio chunks (e.g., vocal segments).

        Args:
            audio_chunks (list): List of 1D numpy arrays or tensors representing audio.

        Returns:
            list of str: Transcribed text for each chunk.
        """

        # Preprocess audio chunks
        processed_inputs = self.processor(
            audio_chunks,
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            return_attention_mask=True,
        )
        if processed_inputs["input_features"].shape[-1] < 3000:
            processed_inputs = self.processor(
                audio_chunks,
                return_tensors="pt",
                sampling_rate=16000,
            )
            # print(inputs)
        # print(inputs.keys())

        processed_inputs = processed_inputs.to(self.device, dtype=self.torch_dtype)

        # input_features = processed_inputs["input_features"].to(self.device, dtype=self.torch_dtype)
        # attention_mask = processed_inputs["attention_mask"].to(self.device, dtype=self.torch_dtype)

        gen_kwargs = {
            # "max_new_tokens": 448,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            # "no_speech_threshold": 0.6,
            "return_timestamps": True,
        }

        if translate:
            gen_kwargs["language"] = "en"

        with torch.no_grad():
            start = time.time()
            pred_ids = self.model.generate(
                **processed_inputs,
                # condition_on_prev_tokens=False,
                # temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                # compression_ratio_threshold=1.35,
                # attention_mask=attention_mask,
                **gen_kwargs,
            )
            # attention_mask=processed_inputs["attention_mask"],)
            pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)

        # print(
        # f"\n🕒 Transcription took {time.time() - start:.2f} seconds for {len(audio_chunks)} chunks."
        # )
        cleaned = self.clean_transcription(pred_text)

        # print("📝 Final cleaned transcription:\n", cleaned)
        # print("\n\n📝 Final cleaned transcription:\n", cleaned)
        return cleaned

    def clean_transcription(
        self, pred_text: list[str], repeat_threshold: int = 3
    ) -> str:
        def clean_text(text, to_remove):
            for word in to_remove:
                # Match case-insensitively, but preserve the rest
                text = re.sub(rf"\b{re.escape(word)}\b", "", text, flags=re.IGNORECASE)
            return text.strip()

        def collapse_repeated_words(text: str) -> str:
            # Collapse repeated single words (with optional commas or whitespace)
            pattern = re.compile(
                rf"\b(\w+)([,\s]+(?:\1\b[,\s]+){{{repeat_threshold - 1},}})",
                flags=re.IGNORECASE,
            )
            return pattern.sub(r"\1 ", text)

        to_remove = ["Thank you.", "music"]
        res = [collapse_repeated_words(clean_text(x, to_remove)) for x in pred_text]

        # res2 = [x for x in res if x != "" and x != " " and len(x.split(" ")) > 2]
        # Join lines, remove placeholder phrases
        # text = "\n\n".join(res2)

        # Collapse repetitions
        # cleaned_text = collapse_repeated_words(text)

        return res

    def batch_transcribe_by_song(
        self, audio_chunks_by_song: dict, batch_size: int = 64, translate=False
    ):
        """
        Transcribes audio chunks from multiple songs in batches and returns results grouped by song.

        Args:
            model: Your transcription model with a `.transcribe_chunks()` method.
            audio_chunks_by_song (dict): Dict where keys are song IDs and values are lists of audio chunks (1D numpy arrays or tensors).
            batch_size (int): Number of chunks per batch.
            translate (bool): Whether to translate to English.

        Returns:
            dict: {song_id: [transcription1, transcription2, ...]}
        """
        # Flatten all chunks with song ID tracking
        all_chunks = []
        chunk_sources = []

        for song_id, chunks in audio_chunks_by_song.items():
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_sources.append(song_id)

        # Transcribe in batches
        all_transcriptions = []
        # batch_size = len(all_chunks)
        for i in range(0, len(all_chunks), batch_size):
            logging.info(
                f"Transcribing batch {i // batch_size + 1} of {len(all_chunks) // batch_size + 1}..."
            )
            batch = all_chunks[i : i + batch_size]
            batch_transcriptions = self.transcribe_chunks(batch, translate=translate)

            # If `transcribe_chunks` returns a single string, wrap it
            if isinstance(batch_transcriptions, str):
                batch_transcriptions = [batch_transcriptions]

            all_transcriptions.extend(batch_transcriptions)

        # Group transcriptions back by song
        results_by_song = defaultdict(list)
        for song_id, text in zip(chunk_sources, all_transcriptions):
            if text.strip() != "" and len(text.split(" ")) > 3:
                results_by_song[song_id].append(text)

        results_by_song = {
            k: "\n\n".join(v) for k, v in results_by_song.items() if len(v) > 0
        }

        return dict(results_by_song)