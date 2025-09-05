from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger
from evaluate import load
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Load Word Error Rate (WER) metric once
wer_metric = load("wer")


def calculate_wer(row: pd.Series, col_reference: str, col_transcription: str) -> Optional[float]:
    """
    Compute the Word Error Rate (WER) for a single row.

    Parameters
    ----------
    row : pd.Series
        Row containing reference and transcription columns.
    col_reference : str
        Name of the reference (ground truth) column.
    col_transcription : str
        Name of the predicted transcription column.

    Returns
    -------
    float or None
        WER score (0=perfect, 1=worst) or None if inputs are invalid.
    """
    ref = row.get(col_reference)
    hyp = row.get(col_transcription)

    # Ensure both are strings before stripping/lowercasing
    if not isinstance(ref, str) or not isinstance(hyp, str):
        return None

    ref = ref.strip().lower()
    hyp = hyp.strip().lower()

    return wer_metric.compute(predictions=[hyp], references=[ref])


def compute_correlation(df: pd.DataFrame, col1: str, col2: str) -> Optional[Tuple[float, float]]:
    """
    Pearson correlation between two numeric columns.

    Returns
    -------
    (correlation, p_value) or None if either column is missing.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return None
    # Drop NaNs independently to avoid misalignment issues; align indexes after drop
    s1 = df[col1].dropna()
    s2 = df[col2].dropna()
    common_index = s1.index.intersection(s2.index)
    if len(common_index) == 0:
        return None
    correlation, p_value = pearsonr(s1.loc[common_index], s2.loc[common_index])
    return correlation, p_value


def plot_scatterplot(
    df: pd.DataFrame,
    col_vocalness: str,
    col_wer_1: str,
    col_wer_2: str,
    save_path: Optional[Path] = None,
) -> None:
    """
    Scatter plot of vocalness vs. two WER columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the columns to plot.
    col_vocalness : str
        Column with vocalness scores (x-axis).
    col_wer_1 : str
        First WER column (y-axis).
    col_wer_2 : str
        Second WER column (y-axis).
    save_path : Optional[Path]
        If provided, figure will be saved to this path.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=col_vocalness, y=col_wer_1, label=f"{col_wer_1}", alpha=0.7)
    sns.scatterplot(data=df, x=col_vocalness, y=col_wer_2, label=f"{col_wer_2}", alpha=0.7)
    plt.xlabel("Vocalness Score")
    plt.ylabel("Word Error Rate (WER)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.close()


def compare_transcription_raw_vocal(
    path_metadata: Path,
    col_reference: str,
    col_target1: str,
    col_target2: str,
    col_vocalness: str,
    *,
    scatter_out: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compare two transcription columns to a reference using WER and analyze correlation with vocalness.

    Steps
    -----
    1) Load metadata CSV
    2) Compute WER for each transcription column vs. reference
    3) Log mean/std WER and Pearson correlation with vocalness
    4) Optionally save a scatter plot of vocalness vs. both WER columns

    Parameters
    ----------
    path_metadata : Path
        CSV path with at least the columns provided below.
    col_reference : str
        Ground-truth reference text column.
    col_target1 : str
        First transcription column.
    col_target2 : str
        Second transcription column.
    col_vocalness : str
        Numeric vocalness score column.
    scatter_out : Optional[Path]
        If provided, saves a scatter plot image to this path.

    Returns
    -------
    pd.DataFrame
        The input DataFrame augmented with `wer_<col_target1>` and `wer_<col_target2>`.
    """
    df = pd.read_csv(path_metadata)

    # Basic checks
    required = {col_reference, col_target1, col_target2, col_vocalness}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Report vocalness stats
    logger.info(f"Vocalness: {df[col_vocalness].mean():.4f} (std: {df[col_vocalness].std():.4f})")

    # Compute WER and correlation for each target
    for col_target in (col_target1, col_target2):
        wer_col = f"wer_{col_target}"
        df[wer_col] = df.apply(lambda row: calculate_wer(row, col_reference, col_target), axis=1)

        mean_wer = df[wer_col].mean()
        std_wer = df[wer_col].std()
        logger.info(f"[{col_target}] WER: {mean_wer:.4f} (std: {std_wer:.4f})")

        corr = compute_correlation(df, col_vocalness, wer_col)
        if corr is not None:
            correlation, p_value = corr
            logger.info(f"[{col_target}] Pearson corr(vocalness, {wer_col}): {correlation:.4f} (p={p_value:.2e})")
        else:
            logger.warning(f"[{col_target}] Could not compute correlation (missing/empty columns).")

    # Optional scatter plot
    plot_scatterplot(df, col_vocalness, f"wer_{col_target1}", f"wer_{col_target2}", save_path=scatter_out)

    return df
