
import pandas as pd
from src.instrumentalvocal.main import InstrumentalDetector
from tqdm import tqdm
import os
import logging

dataset_name = "shs100k"
batch_num = 0 # 1...12

# Load dataset 
df = pd.read_csv(f"src/data/benchmarks/SHS100K-RADAR.csv") #{dataset_name}.csv")
init_len = len(df)

# Sort and slice the dataset for processing
df = df.sort_values(by="md5_encoded").reset_index(drop=True)
df = df.iloc[batch_num * 10_000:(batch_num + 1) * 10_000]
print(f"Processing batch {batch_num + 1} of {len(df)} ({init_len}) for dataset {dataset_name}")

# Initialize the InstrumentalDetector
detector = InstrumentalDetector()

# Loop over all audio files in the dataset and detect instrumental segments
results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running Instrumental Detector"):
    try:
        #path_audio = f"src/data/audio/{dataset_name}/{row['md5_encoded']}.mp3"
        path_audio = f"/nfs/interns/jaffolter/multi-view-ssl-benchmark/src/benchmarks/shs100k_audio/{row['md5_encoded']}.mp3"
        if not os.path.exists(path_audio):
            logging.warning(f"Audio file {path_audio} does not exist. Skipping.")
            continue

        res = detector.detect(path_audio)
        results.append(
            {
                "md5_encoded": row["md5_encoded"],
                "is_instrumental": res["is_instrumental"],
                "all_instrumental": res["all_instrumental"],
                "vocalness_score": res["vocalness_score"],
                "res_detection": res["segments"],
                "vocal_segments": res["vocal_segments"],
            }
        )

    except Exception as e:
        logging.error(f"❌ Error processing {row['file']}: {e}")
        continue

# Save results 
res_df = pd.DataFrame(results)
res_df.to_csv(f"src/data/benchmarks/{dataset_name}_instrumental_detection_{batch_num}.csv", index=False)
