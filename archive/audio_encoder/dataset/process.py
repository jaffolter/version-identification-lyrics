import pandas as pd
import ast


def process_df(batch, idx, df_metadata):
    """
    Process a batch of data.
    """
    # Load transcriptions
    df = pd.read_csv(f"data/csv/discogs_vi_{batch}_{idx}.csv", engine="python", on_bad_lines="skip")
    print(df.head(2)) 
    if batch == 0:
        df = df.rename(columns={"transcription": "transcription_vocal"})

    df["md5_encoded"] = df["file"].apply(lambda x: x.split(".")[0])

    # Filter out rows non-present in df
    df_metadata = df_metadata[df_metadata["md5_encoded"].isin(df["md5_encoded"])]
    df_metadata = df_metadata.merge(df, on="md5_encoded", how="left")

    # Load vocalinstrumental segments
    df_vocalinstrumental = pd.read_csv(f"data/discogs_vi_full_{batch}_{idx}.csv", engine="python", on_bad_lines="skip")
    print(df_vocalinstrumental.head(2))
    df_vocalinstrumental["md5_encoded"] = df_vocalinstrumental["file"].apply(lambda x: x.split(".")[0])

    df_metadata = df_metadata.merge(
        df_vocalinstrumental[["md5_encoded", "vocal_segments", "vocalness_score"]], on="md5_encoded", how="left"
    )
    df_metadata = df_metadata.drop(columns=["file"])

    df_metadata = df_metadata.drop_duplicates(subset=["version_id"])
    # print(df_metadata)

    # Remove nan on transcription_vocal or vocal_segments
    print(len(df_metadata))

    df_metadata = df_metadata[df_metadata["transcription_vocal"].notna() & df_metadata["vocal_segments"].notna()]

    print(len(df_metadata))

    return df_metadata


def get_audio_chunks(vocal_segments):
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
            chunks.append((start, start + 30, 30))
            start += 30
            seg_dur = end - start
            
        chunk = chunks.append((start, end, end - start))

    return chunks


# Load metadata
"""
df_metadata = pd.read_csv("src/benchmarks/discogs_vi_mini.csv")
print(df_metadata.head(2))

chunks_timestamps = []
chunk_df = []

for _, row in df_metadata.iterrows():
    print("----------------------")
    t = row["transcription_vocal"]
    t_segments = t.split("\n\n")   
    
    vocal_segments = ast.literal_eval(row["vocal_segments"])
    chunks = get_audio_chunks(vocal_segments)
    nb_chunks = len(chunks)
    nb_t = len(t_segments)
    
    if nb_chunks > nb_t:
        chunks = sorted(chunks, key=lambda x: x[2], reverse=True)
        chunks = chunks[:nb_t]
        chunks = sorted(chunks, key=lambda x: x[0], reverse=False)

    print(chunks)

        #df_chunks = pd.DataFrame(chunk_df)
"""
    
df_metadata = pd.read_csv("full_metadata.csv")
final_dfs = []

for batch in range(7):
    for idx in range(1, 3):
        if batch in [0, 6] and idx == 2:
            continue

        print(f"Processing batch {batch}, idx {idx}")

        # Merge
        df = process_df(batch, idx, df_metadata)
        print("------------")
        print(df.head(2))

        chunks_timestamps = []

        chunk_df = []
        for _, row in df.iterrows():
            t = row["transcription_vocal"]
            t_segments = t.split("\n\n")
            vocal_segments = ast.literal_eval(row["vocal_segments"])

            chunks = get_audio_chunks(vocal_segments)

            nb_chunks = len(chunks)
            nb_t = len(t_segments)

            if nb_chunks > nb_t:
                chunks = sorted(chunks, key=lambda x: x[2], reverse=True)
                chunks = chunks[:nb_t]
                chunks = sorted(chunks, key=lambda x: x[0], reverse=False)

            chunks_timestamps.append(chunks)

            for i, chunk in enumerate(chunks):
                chunk_df.append(
                    {
                        "version_id": row["version_id"],
                        "clique_id": row["clique_id"],
                        "md5_encoded": row["md5_encoded"],
                        "deezer_id": row["deezer_id"],
                        "chunk_id": i,
                        "start": chunk[0],
                        "end": chunk[1],
                        "duration": chunk[2],
                        "transcription": t_segments[i],
                    }
                )

        df["chunks_timestamps"] = chunks_timestamps
        # print(df)

        df_chunks = pd.DataFrame(chunk_df)
        print(df_chunks)

        final_dfs.append(df_chunks)

        # df.to_csv("custom_model/data/metadata_mini.csv", index=False)
        # df_chunks.to_csv("custom_model/data/metadata_mini_chunks.csv", index=False)


final_df = pd.concat(final_dfs, ignore_index=True)
print(final_df)
final_df.to_csv("data/metadata_final.csv", index=False)
"""