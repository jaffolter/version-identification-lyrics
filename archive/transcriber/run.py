from src.transcriber.main import Transcriber
import os
import pandas as pd
import ast
from tqdm import tqdm
import logging
import torch
from loguru import logger


# ---- Configuration ----
dataset_name = "shs100k"
audio_folder = "/nfs/interns/jaffolter/multi-view-ssl-benchmark/src/benchmarks/shs100k_audio/"  # f"src/data/audio/{dataset_name}"        # Folder containing audio files

batch_size_songs = 32  # Number of songs to process in one batch
batch_size_whisper = 64  # Number of audio chunks to transcribe in one batch (Whisper parameter)

vocal = True  # If True, will use vocal segments for transcription (obtained via instrumentalvocal)
translate = False  # If True, will translate transcriptions to English (Whisper parameter)
name_col = "transcription_vocal"  # Name of the column to store transcriptions in the output CSV (transcription_vocal or transcription_vocal_english, transcription)

batch_num = 0  # Batch number to process
output_csv = f"src/data/benchmarks/{dataset_name}_transcription_{batch_num}.csv"  # Output CSV file for transcriptions

# ---- Setup ----
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Intialize subset of data to transcribe
# df = pd.read_csv(f"src/data/benchmarks/{dataset_name}_vocal.csv", engine="python", on_bad_lines="skip")
df = pd.read_csv(
    f"src/data/benchmarks/{dataset_name}_instrumental_detection_0.csv", engine="python", on_bad_lines="skip"
)

init_len = len(df)
df = df.sort_values(by="md5_encoded").reset_index(drop=True)
df = df.iloc[batch_num * 10_000 : (batch_num + 1) * 10_000]
print(f"Processing batch {batch_num + 1} of {len(df)} ({init_len}) for dataset {dataset_name}")

# Check if output CSV already exists and filter out already processed files
if os.path.exists(output_csv):
    existing = pd.read_csv(output_csv)
    done_files = set(existing["md5_encoded"])
    print(f"📝 Found existing output: {len(done_files)} transcriptions already done.")
    df = df[~df["md5_encoded"].isin(done_files)]  # Filter out already processed
    transcriptions_df = existing
else:
    done_files = set()
    transcriptions_df = pd.DataFrame(columns=["md5_encoded", name_col])

# If no files left to process, exit
if df.empty:
    print("✅ All files already transcribed!")
    exit()


# ---- Init model ----
t = Transcriber("openai/whisper-large-v3-turbo", dataset_name)

chunk_dict = {}
current_batch = 0

all_transcriptions = []
# ---- Process ----
for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
    if idx % 100 == 0:
        logger.info(f"Processing file {idx + 1}/{len(df)}")
    try:
        # Load audio file
        audio_path = os.path.join(audio_folder, f"{row['md5_encoded']}.mp3")

        # Split audio into chunks if not using vocal segments, otherwise use vocal segments and
        # ensure we have 30 seconds of audio for transcription
        if not vocal:
            chunks = t.split_audio(audio_path)
        else:
            vocal_segments = ast.literal_eval(row["vocal_segments"])
            chunks = t.split_vocal_segments(audio_path, vocal_segments)

        if len(chunks) > 0:
            chunk_dict[row["md5_encoded"]] = chunks

        # We have enough chunks to process (batch_size_songs), so we can transcribe them
        if len(chunk_dict) >= batch_size_songs:
            batch_results = t.batch_transcribe_by_song(chunk_dict, batch_size=batch_size_whisper, translate=translate)

            # Retrieve results and update transcriptions DataFrame
            batch_rows = [{"md5_encoded": f, name_col: txt} for f, txt in batch_results.items()]

            batch_df = pd.DataFrame(batch_rows)

            transcriptions_df = pd.concat([transcriptions_df, batch_df], ignore_index=True)

            # Save updated file
            transcriptions_df.to_csv(output_csv, index=False)
            logger.info(f"💾 Updated saved to {output_csv}")

            chunk_dict = {}
            current_batch += 1

    except Exception as e:
        logger.error("Issue : ", e)
        continue

# ---- Final batch ----
if chunk_dict:
    batch_results = t.batch_transcribe_by_song(chunk_dict, batch_size=batch_size_whisper, translate=False)

    batch_rows = [{"md5_encoded": f, name_col: txt} for f, txt in batch_results.items()]
    batch_df = pd.DataFrame(batch_rows)
    transcriptions_df = pd.concat([transcriptions_df, batch_df], ignore_index=True)
    transcriptions_df.to_csv(output_csv, index=False)
    logger.info(f"✅ Final update saved to {output_csv}")
