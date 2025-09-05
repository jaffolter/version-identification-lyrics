#!/usr/bin/env python3
"""
Created on Wed May 22 19:27:14 2024
@author: Alan Ng

Command-line utility to train a production-ready model by learning all
available works and perfs. Uses a stratified K-fold split to handle realistic
data distributions in CSI data sets.

Expects hyperparameters in a hparams_prod.yaml file which expects all the same
hyperparameters as documented for the training hparams.yaml, except:

    Set train_path to point to your full dataset, such as the full.txt output 
    from tools/extract_csi_features.py
    
    val_path will be ignored, since validation sets will be generated for each
    fold from the train_path dataset.
    
    test_path will only be used during the final full-dataset training run,
    as its validation set.
    
    You may also specify one or more external testsets, which will be used
    normally during training for Tensorboard display.

    Set a lower early_stopping_patience than you used in research mode to avoid 
    overfitting in each fold.
    
    See additional hyperparameters described in hparams_prod.yaml in the section
    "Training parameters only used by train_prod"

Example launch command:
python -m tools.train_prod training/yourdata

The required model_dir parameter is the relative path where this script
creates a subfolder "prod_checkpoints" containing checkpoint files.

Optionally specify in MAP_TESTSETS which testsets to use to define peak mAP.
Add or remove from this list based on your relevant testsets.
This allows train_prod to optimize quality of training by
selecting the best epoch (defined by recent peak mAP in your selected testsets)
to return to before starting the next fold. Note: Setting testsets here does
mean that checkpoint data for epochs after the best epoch in each fold
will be deleted.

"""

MAP_TESTSETS = ["reels50easy", "reels50hard", "reels50transpose"]

import argparse
import os, glob, shutil
import time
import sys
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.model_selection import StratifiedKFold
from CoverHunterMPS.src.trainer import Trainer
from CoverHunterMPS.src.model import Model
from CoverHunterMPS.src.utils import create_logger, get_hparams_as_string, load_hparams
from CoverHunterMPS.src.dataset import AudioFeatDataset, read_lines
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def extract_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        labels.append(label.item())
    return np.array(labels)


def write_temp_file(data_lines, indices, filepath):
    with open(filepath, "w") as f:
        for idx in indices:
            f.write(data_lines[idx] + "\n")


def save_fold_indices(fold_indices, filepath):
    with open(filepath, "w") as f:
        json.dump(fold_indices, f)


def load_fold_indices(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def get_map_from_logs(log_dir, testset_name, epoch):
    """
    Get mAP value for specific testset and epoch
    """
    event_files = sorted(
        [f for f in os.listdir(log_dir) if "events.out.tfevents" in f],
        key=lambda x: os.path.getctime(os.path.join(log_dir, x)),
    )
    if not event_files:
        return None

    event_file = os.path.join(log_dir, event_files[-1])
    ea = EventAccumulator(event_file)
    ea.Reload()

    try:
        map_values = ea.Scalars(f"mAP/{testset_name}")
        # Find the closest event to our target epoch
        closest_event = min(
            map_values, key=lambda x: abs(x.step - epoch), default=None
        )
        # Allow events within 1 step of target epoch
        if closest_event and abs(closest_event.step - epoch) <= 1:
            return closest_event.value
    except KeyError:
        return None
    return None


def get_average_map_for_epoch(log_dir, testsets, epoch):
    """
    Calculate average mAP across all specified testsets for a given epoch
    """
    maps = []
    for testset in testsets:
        map_value = get_map_from_logs(log_dir, testset, epoch)
        if map_value is not None:
            maps.append(map_value)

    return np.mean(maps) if maps else None


def find_peak_map_epoch(
    log_dir, testsets, current_epoch, early_stopping_window
):
    """
    Find the epoch with highest average mAP within the early stopping window

    Args:
        log_dir: Directory containing tensorboard logs
        testsets: List of testset names to check
        current_epoch: The epoch at which training stopped
        early_stopping_window: Number of epochs to look back
    """
    event_files = sorted(
        [f for f in os.listdir(log_dir) if "events.out.tfevents" in f],
        key=lambda x: os.path.getctime(os.path.join(log_dir, x)),
    )
    if not event_files:
        return None

    event_file = os.path.join(log_dir, event_files[-1])
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Calculate the earliest epoch to consider
    earliest_epoch = max(0, current_epoch - early_stopping_window)

    # Get all epochs where we have mAP values, filtered by window
    epochs_to_check = set()
    for testset in testsets:
        try:
            events = ea.Scalars(f"mAP/{testset}")
            epochs_to_check.update(
                event.step
                for event in events
                if earliest_epoch <= event.step <= current_epoch
            )
        except KeyError:
            continue

    if not epochs_to_check:
        return None

    # Find epoch with highest average mAP within window
    best_epoch = None
    best_map = -float("inf")
    for epoch in epochs_to_check:
        avg_map = get_average_map_for_epoch(log_dir, testsets, epoch)
        if avg_map is not None and avg_map > best_map:
            best_map = avg_map
            best_epoch = epoch

    return best_epoch


def cleanup_checkpoints(checkpoint_dir, best_epoch):
    """
    Remove all checkpoints and testset embeddings after the best epoch
    """
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(("g_", "do_")):
            epoch = int(filename.split("_")[1])
            if epoch > best_epoch:
                os.remove(os.path.join(checkpoint_dir, filename))
    embed_dirs = glob.glob(
        os.path.join(os.path.dirname(checkpoint_dir), "embed_*_*")
    )
    for directory in embed_dirs:
        shutil.rmtree(directory)


def cross_validate(
    hp, model_class, device, checkpoint_dir, model_dir, run_id, n_splits=5
):
    """
    Perform k-fold cross-validation to train on entire dataset
    for production use.
    """
    logger = create_logger()

    # Read data lines from the original file
    data_lines = read_lines(hp["train_path"])

    # Load the dataset once for splitting purposes
    initial_dataset = AudioFeatDataset(
        hp,
        data_path=None,
        data_lines=data_lines,
        train=False,  # No augmentation at this point
        mode=hp["mode"],
        chunk_len=hp["chunk_frame"][0] * hp["mean_size"],
    )

    # Extract labels
    labels = extract_labels(initial_dataset)

    # Generate indices for stratified k-fold
    indices = np.arange(len(initial_dataset))

    # Save folds in case training needs to be restarted after an interruption
    fold_indices_file = os.path.join(model_dir, "fold_indices.json")
    active_fold_file = os.path.join(model_dir, "active_fold.txt")
    last_completed_fold = -1

    if os.path.exists(active_fold_file):
        with open(active_fold_file, "r") as f:
            last_completed_fold = int(f.readline().strip())
        logger.info(f"Resuming from fold {last_completed_fold + 1}")
        fold_indices = load_fold_indices(fold_indices_file)
    else:
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=hp["seed"]
        )
        fold_indices = [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in kf.split(indices, labels)
        ]
        save_fold_indices(fold_indices, fold_indices_file)

    fold_results = []

    original_train_path = hp["train_path"]
    original_test_path = hp.pop("test_path")  # save for final full dataset

    # Identify available testsets from those specified in MAP_TESTSETS
    available_testsets = [t for t in MAP_TESTSETS if t in hp]
    if not available_testsets:
        logger.warning(
            "No specified testsets found in hyperparameters. Using validation loss for peak detection."
        )

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        if fold <= last_completed_fold:
            continue

        logger.info(f"Training on fold {fold+1}/{n_splits}")

        # Write temporary train and val files
        train_file = os.path.join(model_dir, f"train_fold_{fold+1}.txt")
        val_file = os.path.join(model_dir, f"val_fold_{fold+1}.txt")
        write_temp_file(data_lines, train_idx, train_file)
        write_temp_file(data_lines, val_idx, val_file)
        logger.info(
            f"Temporary fold {fold+1} data files created at {train_file} and {val_file}"
        )

        # Temporarily change hp paths
        hp["train_path"] = train_file
        hp["val_path"] = val_file

        # Create a unique log path for this fold
        log_path = os.path.join(model_dir, "logs", f"{run_id}_fold_{fold+1}")
        os.makedirs(log_path, exist_ok=True)

        # Check if this is a new fold start
        fold_start_file = os.path.join(model_dir, f"fold_{fold+1}_started.txt")
        is_new_fold_start = not os.path.exists(fold_start_file)

        # Instantiate and train a new Trainer instance for this fold
        trainer = Trainer(
            hp=hp,
            model=model_class,
            device=device,
            log_path=log_path,
            checkpoint_dir=checkpoint_dir,
            model_dir=model_dir,
            only_eval=False,
            first_eval=False,
        )
        trainer.configure_optimizer()
        trainer.load_model()
        trainer.configure_scheduler()

        # different learning-rate strategy for all folds after the first
        if fold > 0 and is_new_fold_start:
            # Calculate evenly spaced learning rates from 0.0001 to min_lr
            lr_range = hp["lr_initial"] - hp["min_lr"]
            lr_step = lr_range / (
                n_splits - 1
            )  # -1 since first fold uses original lr
            new_lr = (
                hp["lr_initial"] - (fold - 1) * lr_step
            )  # fold-1 since we start at fold 1
            trainer.reset_learning_rate(new_lr=new_lr)
            hp["lr_decay"] = hp["fold_lr_decay"]
            logger.info(
                f"Adjusted learning rate for fold {fold+1}: lr={new_lr}, min_lr={hp['min_lr']}, decay={hp['lr_decay']}"
            )
        else:
            logger.info(f"Resuming learning rate for fold {fold+1}")

        # Mark this fold as started
        with open(fold_start_file, "w") as f:
            f.write(f"Fold {fold+1} started")

        trainer.train(max_epochs=500)

        # Find peak mAP epoch achieved so far
        # After trainer.train() completes for each fold:
        trainer.summary_writer.flush()
        trainer.summary_writer.close()
        time.sleep(1)  # give OS time to save log file
        best_epoch = find_peak_map_epoch(
            log_path,
            available_testsets,
            trainer.epoch,  # Current epoch when training stopped
            hp["early_stopping_patience"] + 1,  # Look back this many epochs
        )
        if best_epoch is not None:
            logger.info(f"Peak average mAP achieved at epoch {best_epoch}")
            cleanup_checkpoints(checkpoint_dir, best_epoch)
        else:
            logger.warning("Could not determine peak mAP epoch")

        fold_results.append(
            {
                "fold": fold,
                "best_validation_loss": trainer.best_validation_loss,
            }
        )

        # Save the last completed fold
        with open(active_fold_file, "w") as f:
            f.write(str(fold))

    logger.info("Cross-validation completed")

    # Train on the full dataset, and use a testset as the validation set
    logger.info("Training on the full dataset")
    hp["train_path"] = original_train_path
    hp["val_path"] = original_test_path

    # Create a unique log path for the full dataset training
    log_path = os.path.join(model_dir, "logs", f"{run_id}_full")
    os.makedirs(log_path, exist_ok=True)

    # Check if this is a new full dataset training start
    full_start_file = os.path.join(model_dir, "full_dataset_started.txt")
    is_new_full_start = not os.path.exists(full_start_file)

    # Instantiate and train a new Trainer instance for the full dataset
    full_trainer = Trainer(
        hp=hp,
        model=model_class,
        device=device,
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        model_dir=model_dir,
        only_eval=False,
        first_eval=False,
    )
    full_trainer.configure_optimizer()
    full_trainer.load_model()
    full_trainer.configure_scheduler()

    # Adjust learning rate for full dataset training if it's a new start
    # let full dataset have a little more learning rate than the final folds did
    if is_new_full_start:
        # recalculate in case resuming after an interruption
        lr_range = hp["lr_initial"] - hp["min_lr"]
        lr_step = lr_range / (
            n_splits - 1
        )  # -1 since first fold uses original lr
        new_lr = hp["min_lr"] + lr_step * hp["full_dataset_lr_boost"]
        hp["lr_decay"] = hp["final_lr_decay"]
        logger.info(
            f"Adjusted learning rate for fold {fold+1}: lr={new_lr}, min_lr={hp['min_lr']}, decay={hp['lr_decay']}"
        )
        full_trainer.reset_learning_rate(new_lr=new_lr)
    else:
        logger.info("Resuming learning rate for full dataset training")

    # allow user to define longer patience for this final run
    if "final_early_stopping_patience" in hp:
        hp["early_stopping_patience"] = hp["final_early_stopping_patience"]

    # Mark full dataset training as started
    with open(full_start_file, "w") as f:
        f.write("Full dataset training started")

    full_trainer.train(max_epochs=500)
    fold_results.append(
        {
            "fold": "full",
            "best_validation_loss": full_trainer.best_validation_loss,
        }
    )
    logger.info("Full dataset training completed")

    return fold_results


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train: python3 -m tools.train_prod model_dir",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_dir")
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="give more debug log",
    )
    parser.add_argument(
        "--runid",
        default="",
        action="store",
        help="put TensorBoard logs in these subfolders of ../logs/ one per fold",
    )

    args = parser.parse_args()
    model_dir = args.model_dir
    run_id = args.runid

    logger = create_logger()
    hp = load_hparams(os.path.join(model_dir, "config/hparams_prod.yaml"))
    match hp["device"]:  # noqa requires python 3.10
        case "mps":
            if not torch.backends.mps.is_available():
                logger.error(
                    "You requested 'mps' device in your hyperparameters"
                    "but you are not running on an Apple M-series chip or "
                    "have not compiled PyTorch for MPS support."
                )
                sys.exit()
            device = torch.device("mps")
            # set multiprocessing method because 'fork'
            # has significant performance boost on MPS vs. default 'spawn'
            mp.set_start_method("fork")
        case "cuda":
            if not torch.cuda.is_available():
                logger.error(
                    "You requested 'cuda' device in your hyperparameters"
                    "but you do not have a CUDA-compatible GPU available."
                )
                sys.exit()
            device = torch.device("cuda")
            mp.set_start_method("spawn")

        case _:
            logger.error(
                "You set device: %s"
                " in your hyperparameters but that is not a valid option or is an untested option.",
                hp["device"],
            )
            sys.exit()

    # Validate testset specifications
    available_testsets = [t for t in MAP_TESTSETS if t in hp]
    if not available_testsets:
        logger.warning(
            "None of the specified MAP_TESTSETS found in hyperparameters. "
            "The last checkpoint after early stopping will be used instead."
        )
    else:
        logger.info(
            f"Using testsets for peak mAP tracking: {available_testsets}"
        )

    logger.info("%s", get_hparams_as_string(hp))

    torch.manual_seed(hp["seed"])

    checkpoint_dir = os.path.join(model_dir, "prod_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    fold_results = cross_validate(
        hp=hp,
        model_class=Model,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_dir=model_dir,
        run_id=run_id,
        n_splits=hp["k_folds"],
    )

    for result in fold_results:
        logger.info(
            f"Fold {result['fold']} - Best Validation Loss: {result['best_validation_loss']}"
        )


if __name__ == "__main__":
    _main()
