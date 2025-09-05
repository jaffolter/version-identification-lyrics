import pandas as pd


def parse_sparse(df, init_df):
    col_to_keep = ["model_name", "MR1", "HR1", "HR10", "MAP10"]
    df = df[col_to_keep]
    print(df)

    init_df = init_df[col_to_keep]
    init_df = init_df.rename(
        columns={
            "MR1": "MR1_init",
            "HR1": "HR1_init",
            "HR10": "HR10_init",
            "MAP10": "MAP10_init",
        }
    )

    df["bge_mean"] = df["model_name"].apply(lambda x: x.endswith("bge-m3+mean"))
    df["bge_max"] = df["model_name"].apply(lambda x: x.endswith("bge-m3+max"))
    df["bm25_mean"] = df["model_name"].apply(lambda x: x.endswith("BM25+mean"))
    df["bm25_max"] = df["model_name"].apply(lambda x: x.endswith("BM25+max"))

    df["model_name_new"] = df["model_name"].apply(
        lambda x: x.replace("+BM25+mean", "")
        .replace("+BM25+max", "")
        .replace("+BAAI/bge-m3+mean", "")
        .replace("+BAAI/bge-m3+max", "")
    )

    df = df.merge(init_df, left_on="model_name_new", right_on="model_name", how="left")

    df["MR1_diff"] = df["MR1"] - df["MR1_init"]
    df["HR1_diff"] = df["HR1"] - df["HR1_init"]
    df["HR10_diff"] = df["HR10"] - df["HR10_init"]
    df["MAP10_diff"] = df["MAP10"] - df["MAP10_init"]

    for col, col2 in zip(
        ["MR1", "HR1", "HR10", "MAP10"],
        ["MR1_diff", "HR1_diff", "HR10_diff", "MAP10_diff"],
    ):
        df[col] = df.apply(lambda x: f"{x[col]:.3f} ({x[col2]:.3f})", axis=1)

    df = df.rename(
        columns={
            "model_name_y": "model_name",
        }
    )

    df_individual = df[
        df["model_name_x"].isin(["BAAI/bge-m3", "BM25", "BAAI/bge-m3-full"])
    ]

    df_bge_mean = df[df["bge_mean"]].copy()
    df_bge_max = df[df["bge_max"]].copy()
    df_bm25_mean = df[df["bm25_mean"]].copy()
    df_bm25_max = df[df["bm25_max"]].copy()

    cols_to_keep = [
        "model_name",
        "MR1",
        "HR1",
        "HR10",
        "MAP10",
    ]
    cols_to_keep_ind = [
        "model_name_x",
        "MR1",
        "HR1",
        "HR10",
        "MAP10",
    ]
    df_bge_mean = df_bge_mean[cols_to_keep]
    df_bge_max = df_bge_max[cols_to_keep]
    df_bm25_mean = df_bm25_mean[cols_to_keep]
    df_bm25_max = df_bm25_max[cols_to_keep]
    df_individual = df_individual[cols_to_keep_ind]
    df_individual = df_individual.rename(columns={"model_name_x": "model_name"})
    # df_bge = df_bge[cols_to_keep]

    # Remove content in parentheses for df_individual
    for col in ["MR1", "HR1", "HR10", "MAP10"]:
        df_individual[col] = df_individual[col].apply(
            lambda x: x.split(" (")[0] if " (" in x else x
        )

    # df_individual = pd.concat([df_individual, df_bge], ignore_index=True)

    # Now concatenate in a single df, but add a row in between to indicate the type of model
    df_bge_mean["type"] = "BGE Mean"
    df_bge_max["type"] = "BGE Max"
    df_bm25_mean["type"] = "BM25 Mean"
    df_bm25_max["type"] = "BM25 Max"
    df_individual["type"] = "Individual Models"
    df = pd.concat(
        [
            df_bge_mean,
            df_bge_max,
            df_bm25_mean,
            df_bm25_max,
            df_individual,
            # df[~df["bge_mean"] & ~df["bge_max"] & ~df["bm25_mean"] & ~df["bm25_max"]],
        ],
        ignore_index=True,
    )
    df = df.sort_values(by=["type", "HR1"], ascending=False)
    return df


def parse_reranker(df, init_df):
    col_to_keep = ["model_name", "MR1", "HR1", "HR10", "MAP10"]
    df = df[col_to_keep]

    init_df = init_df[col_to_keep]
    init_df = init_df.rename(
        columns={
            "MR1": "MR1_init",
            "HR1": "HR1_init",
            "HR10": "HR10_init",
            "MAP10": "MAP10_init",
        }
    )

    df = df.merge(init_df, on="model_name", how="left")

    df["MR1_diff"] = df["MR1"] - df["MR1_init"]
    df["HR1_diff"] = df["HR1"] - df["HR1_init"]
    df["HR10_diff"] = df["HR10"] - df["HR10_init"]
    df["MAP10_diff"] = df["MAP10"] - df["MAP10_init"]

    for col, col2 in zip(
        ["MR1", "HR1", "HR10", "MAP10"],
        ["MR1_diff", "HR1_diff", "HR10_diff", "MAP10_diff"],
    ):
        df[col] = df.apply(lambda x: f"{x[col]:.3f} ({x[col2]:.3f})", axis=1)

    df = df.rename(
        columns={
            "model_name_y": "model_name",
        }
    )
    cols_to_keep = [
        "model_name",
        "MR1",
        "HR1",
        "HR10",
        "MAP10",
    ]
    df = df[cols_to_keep]
    return df


def parse_audio(df, init_df):
    col_to_keep = ["model_name", "MR1", "HR1", "HR10", "MAP10"]
    df = df[col_to_keep]

    init_df = init_df[col_to_keep]
    init_df = init_df.rename(
        columns={
            "MR1": "MR1_init",
            "HR1": "HR1_init",
            "HR10": "HR10_init",
            "MAP10": "MAP10_init",
        }
    )
    init_df = init_df[init_df["model_name"]=="Alibaba-NLP/gte-multilingual-base"]
    
    # modify the model_name in init_df to match the one in df
    init_df["model_name"] = init_df["model_name"].apply(
        lambda x: x.replace("Alibaba-NLP/gte-multilingual-base", "LIE")
    )
    
    df = df.merge(init_df, on="model_name", how="left")

    df["MR1_diff"] = df["MR1"] - df["MR1_init"]
    df["HR1_diff"] = df["HR1"] - df["HR1_init"]
    df["HR10_diff"] = df["HR10"] - df["HR10_init"]
    df["MAP10_diff"] = df["MAP10"] - df["MAP10_init"]

    for col, col2 in zip(
        ["MR1", "HR1", "HR10", "MAP10"],
        ["MR1_diff", "HR1_diff", "HR10_diff", "MAP10_diff"],
    ):
        df[col] = df.apply(lambda x: f"{x[col]:.3f} ({x[col2]:.3f})", axis=1)

    df = df.rename(
        columns={
            "model_name_y": "model_name",
        }
    )
    cols_to_keep = [
        "model_name",
        "MR1",
        "HR1",
        "HR10",
        "MAP10",
    ]
    df = df[cols_to_keep]
    return df
