import pandas as pd


def parse_sparse(df: pd.DataFrame, init_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare sparse models (BGE + BM25 variants) against an initial/baseline table.

    Steps (logic preserved):
      1) Keep metric columns only.
      2) Create flags for each combo suffix (bge-m3+mean/max, BM25+mean/max).
      3) Strip combo suffixes to get a core 'model_name_new'.
      4) Merge with baseline on that core name.
      5) Compute per-metric diffs, and format as 'value (diff)' strings.
      6) Build 4 blocks (BGE mean, BGE max, BM25 mean, BM25 max) + an 'Individual Models' block.
      7) Concatenate blocks, add 'type' column, sort by ['type','HR1'] (lexicographic).

    Notes:
      - Sorting after turning numbers into strings remains lexicographic, as in your original code.
      - Keeps the same column naming ('model_name_x', 'model_name_y') from the merge path you chose.
    """
    # 1) Keep metric columns
    cols = ["model_name", "MR1", "HR1", "HR10", "MAP10"]
    df = df[cols].copy()
    init_df = init_df[cols].copy()

    # Baseline columns renamed with _init suffix
    init_df = init_df.rename(columns={"MR1": "MR1_init", "HR1": "HR1_init", "HR10": "HR10_init", "MAP10": "MAP10_init"})

    # 2) Flags for model variants
    df["bge_mean"] = df["model_name"].apply(lambda x: str(x).endswith("bge-m3+mean"))
    df["bge_max"] = df["model_name"].apply(lambda x: str(x).endswith("bge-m3+max"))
    df["bm25_mean"] = df["model_name"].apply(lambda x: str(x).endswith("BM25+mean"))
    df["bm25_max"] = df["model_name"].apply(lambda x: str(x).endswith("BM25+max"))

    # 3) Derive a core model name to align with init_df
    df["model_name_new"] = (
        df["model_name"]
        .astype(str)
        .str.replace("+BM25+mean", "", regex=False)
        .str.replace("+BM25+max", "", regex=False)
        .str.replace("+BAAI/bge-m3+mean", "", regex=False)
        .str.replace("+BAAI/bge-m3+max", "", regex=False)
    )

    # 4) Merge with baseline using the derived core name
    df = df.merge(init_df, left_on="model_name_new", right_on="model_name", how="left", suffixes=("_x", "_y"))

    # 5) Compute diffs and format "value (diff)"
    df["MR1_diff"] = df["MR1"] - df["MR1_init"]
    df["HR1_diff"] = df["HR1"] - df["HR1_init"]
    df["HR10_diff"] = df["HR10"] - df["HR10_init"]
    df["MAP10_diff"] = df["MAP10"] - df["MAP10_init"]

    for col, dcol in zip(["MR1", "HR1", "HR10", "MAP10"], ["MR1_diff", "HR1_diff", "HR10_diff", "MAP10_diff"]):
        df[col] = df.apply(lambda r: f"{r[col]:.3f} ({r[dcol]:.3f})", axis=1)

    # For the final tables we prefer a single 'model_name' column
    df = df.rename(columns={"model_name_y": "model_name"})

    # Individual models block (exact filter kept)
    df_individual = df[df["model_name_x"].isin(["BAAI/bge-m3", "BM25", "BAAI/bge-m3-full"])].copy()

    # Variant blocks
    df_bge_mean = df[df["bge_mean"]].copy()
    df_bge_max = df[df["bge_max"]].copy()
    df_bm25_mean = df[df["bm25_mean"]].copy()
    df_bm25_max = df[df["bm25_max"]].copy()

    base_cols = ["model_name", "MR1", "HR1", "HR10", "MAP10"]
    base_cols_ind = ["model_name_x", "MR1", "HR1", "HR10", "MAP10"]

    df_bge_mean = df_bge_mean[base_cols]
    df_bge_max = df_bge_max[base_cols]
    df_bm25_mean = df_bm25_mean[base_cols]
    df_bm25_max = df_bm25_max[base_cols]

    df_individual = df_individual[base_cols_ind].rename(columns={"model_name_x": "model_name"})

    # Remove "(diff)" for individual block (exact logic preserved)
    for mcol in ["MR1", "HR1", "HR10", "MAP10"]:
        df_individual[mcol] = df_individual[mcol].apply(
            lambda s: s.split(" (")[0] if isinstance(s, str) and " (" in s else s
        )

    # Tag blocks and concatenate
    df_bge_mean["type"] = "BGE Mean"
    df_bge_max["type"] = "BGE Max"
    df_bm25_mean["type"] = "BM25 Mean"
    df_bm25_max["type"] = "BM25 Max"
    df_individual["type"] = "Individual Models"

    out = pd.concat(
        [df_bge_mean, df_bge_max, df_bm25_mean, df_bm25_max, df_individual],
        ignore_index=True,
    )

    # Same as your code: sort after string-formatting
    out = out.sort_values(by=["type", "HR1"], ascending=False, kind="mergesort")
    return out


def parse_reranker(df: pd.DataFrame, init_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare reranker models against a baseline table.

    Steps:
      1) Keep metrics.
      2) Merge on exact 'model_name'.
      3) Compute diffs and format "value (diff)" strings.
      4) Keep display columns.
    """
    cols = ["model_name", "MR1", "HR1", "HR10", "MAP10"]
    df = df[cols].copy()
    init_df = init_df[cols].copy()

    init_df = init_df.rename(columns={"MR1": "MR1_init", "HR1": "HR1_init", "HR10": "HR10_init", "MAP10": "MAP10_init"})

    df = df.merge(init_df, on="model_name", how="left")

    df["MR1_diff"] = df["MR1"] - df["MR1_init"]
    df["HR1_diff"] = df["HR1"] - df["HR1_init"]
    df["HR10_diff"] = df["HR10"] - df["HR10_init"]
    df["MAP10_diff"] = df["MAP10"] - df["MAP10_init"]

    for col, dcol in zip(["MR1", "HR1", "HR10", "MAP10"], ["MR1_diff", "HR1_diff", "HR10_diff", "MAP10_diff"]):
        df[col] = df.apply(lambda r: f"{r[col]:.3f} ({r[dcol]:.3f})", axis=1)

    # Keep the columns you display
    cols_out = ["model_name", "MR1", "HR1", "HR10", "MAP10"]
    df = df[cols_out]
    return df


def parse_audio(df: pd.DataFrame, init_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare audio encoder results vs. a specific baseline row
    (baseline filtered to 'Alibaba-NLP/gte-multilingual-base' and renamed to 'LIE').

    Steps:
      1) Keep metrics.
      2) Filter baseline to the single row.
      3) Rename its model_name -> 'LIE' to align with df.
      4) Merge, compute diffs, format "value (diff)" strings.
      5) Keep display columns.
    """
    cols = ["model_name", "MR1", "HR1", "HR10", "MAP10"]
    df = df[cols].copy()
    init_df = init_df[cols].copy()

    init_df = init_df.rename(columns={"MR1": "MR1_init", "HR1": "HR1_init", "HR10": "HR10_init", "MAP10": "MAP10_init"})
    init_df = init_df[init_df["model_name"] == "Alibaba-NLP/gte-multilingual-base"].copy()

    # Align baseline name to target key in `df`
    init_df.loc[:, "model_name"] = init_df["model_name"].str.replace(
        "Alibaba-NLP/gte-multilingual-base", "LIE", regex=False
    )

    df = df.merge(init_df, on="model_name", how="left")

    df["MR1_diff"] = df["MR1"] - df["MR1_init"]
    df["HR1_diff"] = df["HR1"] - df["HR1_init"]
    df["HR10_diff"] = df["HR10"] - df["HR10_init"]
    df["MAP10_diff"] = df["MAP10"] - df["MAP10_init"]

    for col, dcol in zip(["MR1", "HR1", "HR10", "MAP10"], ["MR1_diff", "HR1_diff", "HR10_diff", "MAP10_diff"]):
        df[col] = df.apply(lambda r: f"{r[col]:.3f} ({r[dcol]:.3f})", axis=1)

    cols_out = ["model_name", "MR1", "HR1", "HR10", "MAP10"]
    df = df[cols_out]
    return df
