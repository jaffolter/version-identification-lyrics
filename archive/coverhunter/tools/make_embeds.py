#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to generate reference embeddings for all production-use audio data
available, using the production-ready model you trained using, for example,
tools.train_prod.py. Intended for use by tools.identify.py or other
applications you might create that use your fully trained model.

Example invocation:
    python -m tools.make_embeds data/covers80 training/covers80

Parameters
----------
data_path : string
    Relative path to the project folder containing a full.txt file that
    you generated using tools.extract_csi_features.py, for example the one 
    you used to train your model. These will be the recordings that your
    inference solution will "know."
    Example: "data/covers80"
    
model_path : string
    Relative path to the project folder containing your trained model.
    Example: "training/covers80"
    This script requires reuse of the following files that you used and
    generated during training of your model:
        [model_path]/config/hparams.yaml
        [model_path]/checkpoints/...
        

Output
------
Pickle file of reference embeddings saved to data_path.


Created on Sat Jul 27 10:54:22 2024
@author: alanngnet
"""

import os
import argparse
import pickle
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # displays progress bar

from CoverHunterMPS.src.model import Model
from CoverHunterMPS.src.dataset import AudioFeatDataset
from CoverHunterMPS.src.utils import load_hparams, read_lines, line_to_dict


def generate_embeddings(model, data_loader, device):
    """Generate embeddings for all samples in the data loader."""
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating embeddings"):
            perf, feat, _ = batch
            feat = feat.float().to(device)
            embed, _ = model.inference(feat)
            for i, p in enumerate(perf):
                embeddings[p] = embed[i].cpu().numpy()
    return embeddings


def main(data_path, model_path):
    # Load hyperparameters
    model_hp = load_hparams(os.path.join(model_path, "config/hparams.yaml"))

    # Set up device
    device = torch.device(model_hp["device"])

    # Initialize model
    model = Model(model_hp).to(device)
    checkpoint_dir = os.path.join(model_path, "checkpoints")
    model.load_model_parameters(checkpoint_dir, device=device)

    # Prepare dataset
    # Need to use full.txt because only it has the necessary workid field
    dataset_file = os.path.join(data_path, "full.txt")
    lines = read_lines(dataset_file)
    # Filter out speed-augmented perfs
    lines = [
        line
        for line in lines
        if not re.match(r"sp_[0-9.]+-.+", line_to_dict(line)["perf"])
    ]
    print(f"Loaded {len(lines)} CQT arrays to compute their embeddings.")

    infer_frame = model_hp["chunk_frame"][0] * model_hp["mean_size"]
    dataset = AudioFeatDataset(
        model_hp,
        data_lines=lines,
        train=False,
        mode=model_hp["mode"],
        chunk_len=infer_frame,
    )

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=model_hp["batch_size"],
        shuffle=False,
        num_workers=model_hp["num_workers"],
        pin_memory=True,
    )

    # Generate embeddings
    embeddings = generate_embeddings(model, data_loader, device)

    # Save embeddings
    output_file = os.path.join(data_path, "reference_embeddings.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"Reference embeddings saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reference embeddings for production use."
    )
    parser.add_argument(
        "data_path", help="Path to the data folder containing dataset.txt"
    )
    parser.add_argument(
        "model_path", help="Path to the folder containing the trained model"
    )
    args = parser.parse_args()

    main(args.data_path, args.model_path)
