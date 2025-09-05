"""
Created by @samuel-gauthier and @alanngnet in April-May 2024.

Tool to launch many sequential training runs for the purpose of discovering
optimal hyperparameter settings for a given dataset, aka "hyperparameter tuning."

"""

import glob, os, sys
import shutil, time, gc
import argparse
from datetime import date
import pprint
import torch
import torch.multiprocessing as mp
import numpy as np
import random
from collections import defaultdict
from CoverHunterMPS.src.trainer import Trainer
from CoverHunterMPS.src.model import Model
from CoverHunterMPS.src.utils import load_hparams, create_logger
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def make_deterministic(seed):
    """
    @samuel-gauthier investigated non-deterministic training behavior on
    his CUDA platform which @alanngnet did not observe on his MPS platform.
    @samuel-gauthier reported: "my tests showed no variance with deterministic
    = true and benchmark = false, and variance if deterministic = false or
    benchmark = true". With benchmark = true, "I found myself with 3 or 4
    different results when launching 10 [training runs]." (quoted from
    correspondence with @alanngnet 29 May 2024). He arrived at this function's
    method of ensuring deterministic training on CUDA platforms.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(
    hp_summary,
    checkpoint_dir,
    hp,
    seed,
):
    hp["seed"] = seed
    make_deterministic(seed)
    # set multiprocessing method because 'fork'
    # has significant performance boost on MPS vs. default 'spawn'
    if torch.backends.mps.is_available():
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass
    else:
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    log_path = os.path.join(
        model_dir,
        "logs",
        hp_summary + f"_seed_{seed}",
        today,
    )
    print("===========================================================")
    print(f"Running experiment with seed {seed}, {hp_summary}")
    os.makedirs(log_path, exist_ok=True)
    shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # must clear temp embeddings otherwise they will be reused for testsset metrics
    directories = glob.glob(os.path.join(model_dir, "embed_*_*"))
    for directory in directories:
        shutil.rmtree(directory)
    pprint.pprint(hp)

    t = Trainer(
        hp,
        Model,
        hp["device"],
        log_path,
        checkpoint_dir,
        model_dir,
        only_eval=False,
        first_eval=False,
    )

    t.configure_optimizer()
    t.load_model()
    t.configure_scheduler()
    t.train(max_epochs=hp["max_epochs"])
    # ensure log files are saved before retrieving metrics
    t.summary_writer.close()

    del t.model
    del t
    gc.collect()
    print(f"Completed experiment with seed {seed}")
    time.sleep(1)  # give OS time to save log file
    return log_path


def get_final_metrics_from_logs(log_dir, test_name):
    # Find the latest event file in the log directory
    event_file = max(
        glob.glob(os.path.join(log_dir, "events.out.tfevents.*")),
        key=os.path.getctime,
    )
    # Load the event file
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Extract the final validation loss and mAP
    val_loss = ea.Scalars("csi_val/foc_loss")[-1].value
    mAP = ea.Scalars(f"mAP/{test_name}")[-1].value
    print(f"Final loss {val_loss}, mAP {mAP}")

    return val_loss, mAP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train-tune: python3 -m tools.train-tune model_dir",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("model_dir")
    args = parser.parse_args()
    model_dir = args.model_dir
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    experiments = load_hparams(
        os.path.join(model_dir, "config/hp_tuning.yaml")
    )
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    test_name = experiments["test_name"]
    # ensure at least one seed
    seeds = experiments.get("seeds", [hp["seed"]])
    chunk_frames = experiments["chunk_frames"]
    # ensure at least one mean_size
    mean_sizes = experiments.get("mean_sizes", [hp["mean_size"]])
    m_per_classes = experiments["m_per_classes"]
    spec_augmentations = experiments["spec_augmentations"]
    losses = experiments["losses"]
    learning_rates = experiments["learning_rates"]
    lr_decays = experiments["lr_decays"]
    adam_betas = experiments["adam_betas"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger()
    today = date.today().strftime("%Y-%m-%d")

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
        case "cuda":
            if not torch.cuda.is_available():
                logger.error(
                    "You requested 'cuda' device in your hyperparameters"
                    "but you do not have a CUDA-compatible GPU available."
                )
                sys.exit()
            device = torch.device("cuda")
        case _:
            logger.error(
                "You set device: %s in your hyperparameters but that is "
                "not a valid option or is an untested option.",
                hp["device"],
            )
            sys.exit()

    all_results = {}

    # chunk_frame experiments
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]
    # default 15 max epochs unless specified in hp_tuning.yaml
    hp["max_epochs"] = experiments.get("max_epochs", 15)
    results = defaultdict(list)
    for chunk_frame in chunk_frames:
        hp["chunk_frame"] = chunk_frame
        for mean_size in mean_sizes:
            hp["mean_size"] = mean_size
            hp["chunk_s"] = chunk_frame[0] * mean_size / 25
            for seed in seeds:
                hp_summary = (
                    "chunk_frame"
                    + "_".join([str(c) for c in chunk_frame])
                    + f"_mean_size{mean_size}"
                )
                log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
                final_val_loss, final_map = get_final_metrics_from_logs(
                    log_path, test_name
                )
                results["val_loss"].append(final_val_loss)
                results["map"].append(final_map)
            mean_loss = np.mean(results["val_loss"])
            std_loss = np.std(results["val_loss"])
            mean_map = np.mean(results["map"])
            std_map = np.std(results["map"])
            all_results[hp_summary] = {
                "val_loss": {"mean": mean_loss, "std": std_loss},
                "map": {"mean": mean_map, "std": std_map},
            }
            results.clear()
            print(f"Results for {hp_summary}")
            pprint.pprint(all_results[hp_summary])

    # m_per_class experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]
    hp["max_epochs"] = experiments.get("max_epochs", 15)
    results = defaultdict(list)
    for m_per_class in m_per_classes:
        hp["m_per_class"] = m_per_class
        for seed in seeds:
            hp_summary = f"m_per_class{m_per_class}"
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            final_val_loss, final_map = get_final_metrics_from_logs(
                log_path, test_name
            )
            results["val_loss"].append(final_val_loss)
            results["map"].append(final_map)
        mean_loss = np.mean(results["val_loss"])
        std_loss = np.std(results["val_loss"])
        mean_map = np.mean(results["map"])
        std_map = np.std(results["map"])
        all_results[hp_summary] = {
            "val_loss": {"mean": mean_loss, "std": std_loss},
            "map": {"mean": mean_map, "std": std_map},
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # spec_aug experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]
    hp["max_epochs"] = experiments.get("max_epochs", 15)
    results = defaultdict(list)
    for spec_augmentation in spec_augmentations:
        hp["spec_augmentation"] = spec_augmentation
        random_erase_prob = spec_augmentation["random_erase"]["prob"]
        random_erase_num = spec_augmentation["random_erase"]["erase_num"]
        region_size = spec_augmentation["random_erase"]["region_size"]
        roll_pitch_prob = spec_augmentation["roll_pitch"]["prob"]
        roll_pitch_shift_num = spec_augmentation["roll_pitch"]["shift_num"]
        roll_pitch_method = spec_augmentation["roll_pitch"]["method"]

        for seed in seeds:
            hp_summary = (
                f"erase_prob{random_erase_prob}_num{random_erase_num}_size"
                + "_".join([str(c) for c in region_size])
                + f"_roll_prob{roll_pitch_prob}_shift{roll_pitch_shift_num}"
                + f"_{roll_pitch_method}"
            )
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            final_val_loss, final_map = get_final_metrics_from_logs(
                log_path, test_name
            )
            results["val_loss"].append(final_val_loss)
            results["map"].append(final_map)
        mean_loss = np.mean(results["val_loss"])
        std_loss = np.std(results["val_loss"])
        mean_map = np.mean(results["map"])
        std_map = np.std(results["map"])
        all_results[hp_summary] = {
            "val_loss": {"mean": mean_loss, "std": std_loss},
            "map": {"mean": mean_map, "std": std_map},
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # loss experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]
    hp["max_epochs"] = experiments.get("max_epochs", 15)
    results = defaultdict(list)
    for loss in losses:
        hp["foc"] = foc = loss["foc"]
        foc_dims = foc["output_dims"]
        foc_weight = foc["weight"]
        foc_gamma = foc["gamma"]
        hp["triplet"] = triplet = loss["triplet"]
        triplet_margin = triplet["margin"]
        triplet_weight = triplet["weight"]
        hp["center"] = center = loss["center"]
        center_weight = center["weight"]

        for seed in seeds:
            hp_summary = (
                f"FOC_dims{foc_dims}_wt{foc_weight}_gamma{foc_gamma}_"
                + f"TRIP_marg{triplet_margin}_wt{triplet_weight}_"
                + f"CNTR_wt{center_weight}"
            )
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            final_val_loss, final_map = get_final_metrics_from_logs(
                log_path, test_name
            )
            results["val_loss"].append(final_val_loss)
            results["map"].append(final_map)
        mean_loss = np.mean(results["val_loss"])
        std_loss = np.std(results["val_loss"])
        mean_map = np.mean(results["map"])
        std_map = np.std(results["map"])
        all_results[hp_summary] = {
            "val_loss": {"mean": mean_loss, "std": std_loss},
            "map": {"mean": mean_map, "std": std_map},
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # learning_rate experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]
    hp["max_epochs"] = experiments.get("max_epochs", 15)
    results = defaultdict(list)
    for learning_rate in learning_rates:
        hp["learning_rate"] = learning_rate
        for seed in seeds:
            hp_summary = f"lrate{learning_rate}"
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            final_val_loss, final_map = get_final_metrics_from_logs(
                log_path, test_name
            )
            results["val_loss"].append(final_val_loss)
            results["map"].append(final_map)
        mean_loss = np.mean(results["val_loss"])
        std_loss = np.std(results["val_loss"])
        mean_map = np.mean(results["map"])
        std_map = np.std(results["map"])
        all_results[hp_summary] = {
            "val_loss": {"mean": mean_loss, "std": std_loss},
            "map": {"mean": mean_map, "std": std_map},
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # lr_decay experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]
    hp["max_epochs"] = experiments.get("max_epochs", 15)
    results = defaultdict(list)
    for lr_decay in lr_decays:
        hp["lr_decay"] = lr_decay
        for seed in seeds:
            hp_summary = f"lr_decay{lr_decay}"
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            final_val_loss, final_map = get_final_metrics_from_logs(
                log_path, test_name
            )
            results["val_loss"].append(final_val_loss)
            results["map"].append(final_map)
        mean_loss = np.mean(results["val_loss"])
        std_loss = np.std(results["val_loss"])
        mean_map = np.mean(results["map"])
        std_map = np.std(results["map"])
        all_results[hp_summary] = {
            "val_loss": {"mean": mean_loss, "std": std_loss},
            "map": {"mean": mean_map, "std": std_map},
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    # AdamW betas experiments
    hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
    hp["every_n_epoch_to_save"] = 100
    hp["early_stopping_patience"] = experiments["early_stopping_patience"]
    hp["max_epochs"] = experiments.get("max_epochs", 15)
    results = defaultdict(list)
    for adam_beta in adam_betas:
        hp["adam_b1"] = adam_beta[0]
        hp["adam_b2"] = adam_beta[1]
        for seed in seeds:
            hp_summary = "adam_betas" + "_".join([str(c) for c in adam_beta])
            log_path = run_experiment(hp_summary, checkpoint_dir, hp, seed)
            final_val_loss, final_map = get_final_metrics_from_logs(
                log_path, test_name
            )
            results["val_loss"].append(final_val_loss)
            results["map"].append(final_map)
        mean_loss = np.mean(results["val_loss"])
        std_loss = np.std(results["val_loss"])
        mean_map = np.mean(results["map"])
        std_map = np.std(results["map"])
        all_results[hp_summary] = {
            "val_loss": {"mean": mean_loss, "std": std_loss},
            "map": {"mean": mean_map, "std": std_map},
        }
        results.clear()
        print(f"Results for {hp_summary}")
        pprint.pprint(all_results[hp_summary])

    print("\nSummary of Experiments:")
    for hp_summary, result in all_results.items():
        print(f"\nExperiment: {hp_summary}")
        print(
            f"  Validation Loss: mean = {result['val_loss']['mean']:.4f}, std = {result['val_loss']['std']:.4f}"
        )
        print(
            f"  mAP: mean = {result['map']['mean']:.4f}, std = {result['map']['std']:.4f}"
        )
