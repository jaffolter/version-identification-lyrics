from Transcriber import Transcriber
import os
import pandas as pd
import ast
from tqdm import tqdm
import logging
import torch

# ---- Configuration ----
dataset_name = os.environ.get("DATASET", "discogs_vi")
translate = os.environ.get("TRANSLATE", "True").lower() == "true"
vocal = os.environ.get("VOCAL", "True").lower() == "true"
translate = False
vocal = False

batch_size_songs = 32
batch_size_whisper = 16
audio_folder = f"src/datasets/{dataset_name}/audio"
input_csv = f"src/benchmarks/{dataset_name}_mini.csv"

name_col = (
    "transcription_vocal_english"
    if translate
    else "transcription_vocal"
    if vocal
    else "transcription"
)
output_csv = f"src/benchmarks/discogs_vi_mini_{name_col}.csv"

# ---- Setup ----
print(
    f"Running transcriber benchmark: dataset={dataset_name}, translate={translate}, vocal={vocal}"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---- Load input ----
df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} rows from {input_csv}")

# ---- Load existing output (if any) ----
if os.path.exists(output_csv):
    existing = pd.read_csv(output_csv)
    done_files = set(existing["file"])
    print(f"📝 Found existing output: {len(done_files)} transcriptions already done.")
    df = df[~df["file"].isin(done_files)]  # Filter out already processed
    transcriptions_df = existing
else:
    done_files = set()
    transcriptions_df = pd.DataFrame(columns=["file", name_col])

if df.empty:
    print("✅ All files already transcribed!")
    exit()

# ---- Init model ----
t = Transcriber("openai/whisper-large-v3-turbo", dataset_name)
t.init_model("openai/whisper-large-v3-turbo")

chunk_dict = {}
current_batch = 0
all_transcriptions = []

# ---- Process ----
for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
    try:
        audio_path = os.path.join(audio_folder, row["file"])
        if not vocal:
            chunks = t.split_audio(audio_path)
        else:
            vocal_segments = ast.literal_eval(row["vocal_segments"])
            chunks = t.split_vocal_segments(audio_path, vocal_segments)

        if len(chunks) > 0:
            chunk_dict[row["file"]] = chunks

        if len(chunk_dict) >= batch_size_songs:
            print(f"🌀 Transcribing batch #{current_batch} ({len(chunk_dict)} songs)")
            batch_results = t.batch_transcribe_by_song(
                chunk_dict, batch_size=batch_size_whisper, translate=translate
            )

            batch_rows = [
                {"file": f, name_col: txt} for f, txt in batch_results.items()
            ]
            batch_df = pd.DataFrame(batch_rows)
            transcriptions_df = pd.concat(
                [transcriptions_df, batch_df], ignore_index=True
            )

            # Save updated file
            transcriptions_df.to_csv(output_csv, index=False)
            print(f"💾 Updated saved to {output_csv}")

            chunk_dict = {}
            current_batch += 1

    except Exception as e:
        logging.error(f"⚠️ Error in row {idx} / file {row['file']}: {e}")
        continue

# ---- Final batch ----
if chunk_dict:
    print(f"🌀 Transcribing final batch #{current_batch} ({len(chunk_dict)} songs)")
    batch_results = t.batch_transcribe_by_song(
        chunk_dict, batch_size=batch_size_whisper, translate=translate
    )

    batch_rows = [{"file": f, name_col: txt} for f, txt in batch_results.items()]
    batch_df = pd.DataFrame(batch_rows)
    transcriptions_df = pd.concat([transcriptions_df, batch_df], ignore_index=True)
    transcriptions_df.to_csv(output_csv, index=False)
    print(f"✅ Final update saved to {output_csv}")
