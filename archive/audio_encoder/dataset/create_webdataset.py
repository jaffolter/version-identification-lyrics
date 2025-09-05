import argparse
import sys
from pathlib import Path

import pandas as pd
import webdataset as wds
from tqdm import tqdm


def get_path(root: Path, uid: str, data_type: str) -> Path:
    """
    Construct the path to the .npy file based on data type.

    Args:
        root (Path): Root directory.
        uid (str): Unique file identifier.
        data_type (str): Type of data (e.g., "new_features" or "embeddings").

    Returns:
        Path: Full path to the .npy file.
    """
    if "new_features" in data_type:
        new_uid = uid.replace(".npy", "")
        new_uid += new_uid[-1] + ".npy"
    else:
        new_uid = uid
    return root / new_uid


def main(
    split: str,
    out: str,
    features_root: str = "data/new_features",
    embeddings_root: str = "data/embeddings",
    shard_size: int = 1000,
):
    """
    Main function to create sharded WebDataset archives.

    Args:
        split (str): Dataset split to process (train/val/test).
        out (str): Output path pattern for shards (data/{split}/shard-%06d.tar).
        features_root (str): Root path to audio features.
        embeddings_root (str): Root path to text embeddings.
        shard_size (int): Max samples per shard.
    """
    df = pd.read_csv(f"data/{split}_metadata.csv").sort_values("filename")
    ids = df["filename"].tolist()

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    missing, kept = 0, 0
    with wds.ShardWriter(out, maxcount=shard_size) as sink:
        for uid in tqdm(ids, desc=f"Writing {split} shards"):
            feat_path = Path(features_root) / uid
            emb_path = Path(embeddings_root) / uid

            if not (feat_path.exists() and emb_path.exists()):
                missing += 1
                continue

            sample = {
                "__key__": uid.split("/")[1].replace(".npy", ""),
                "features.npy": feat_path.read_bytes(),
                "text.npy": emb_path.read_bytes(),
            }

            sink.write(sample)
            kept += 1

    print(f"✅ Done: {kept} samples written, {missing} skipped.", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--features-root", type=str, default="data/new_features")
    parser.add_argument("--embeddings-root", type=str, default="data/embeddings")
    parser.add_argument("--out", type=str, default="data/train/shard-%06d.tar")
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()

    main(
        split=args.split,
        features_root=args.features_root,
        embeddings_root=args.embeddings_root,
        out=args.out,
        shard_size=args.shard_size,
    )
