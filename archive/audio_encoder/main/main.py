from loguru import logger

from src.evaluation.main import ModelEvaluation

# Instantiate ModelEvaluation class ---------------------------
evaluator = ModelEvaluation(
    checkpoint_path="src/checkpoints/upbeat-moon-263/attention_mse_whisper_frozen_True_epoch_2.pth",
    config_path="src/conf/config_best.yaml",
    session_name="upbeat_moon_263"
)


# Compute embeddings for benchmarks -----------------------------------------------
"""benchmarks = ['shs100k'] #['covers80', 'shs100k', 'discogs_vi_mini', 'discogs_vi']
for dataset_name in benchmarks:
    logger.info(f"Computing embeddings for dataset: {dataset_name}")
    evaluator.compute_embeddings_benchmarks(
        dataset_name=dataset_name,
        path_metadata=f"src/benchmarks/{dataset_name}ok.csv",
        path_audio_folder=f"src/benchmarks/{dataset_name}_audio" if 'mini' not in dataset_name else "src/data/audio/audio",
        compute_vocal_segments=False,       # Set to False for benchmarks since vocal segments are already pre-computed and stored in the dataframe
        get_single_embedding=True
    )"""


# Evaluation of benchmarks -----------------------------------------
for dataset_name in ['shs100k']:
    for detail in ['q1', 'median', 'q3']:
        logger.info(f"Evaluating version identification for dataset: {dataset_name}, detail: {detail}")
        evaluator.evaluate_version_identification(dataset_name=dataset_name, details=detail)


""" 
# Compute embeddings for test set -----------------------------------------------
test_dataloader = evaluator.load_test_set()
logger.info("Computing embeddings for test set")
evaluator.compute_embeddings_from_test_set(
    dataloader=test_dataloader,
    get_single_embedding=False,
    compute_text_embeddings=True  
)

# Evaluation of test set -------------------------------------------

# 1. PCA between audio embeddings and text embeddings 
n = 100  # Number of samples to plot
logger.info(f"Performing PCA on {n} samples of audio and text embeddings")
text_embeddings, audio_embeddings = evaluator.load_local_embeddings_test_set(n=n)
evaluator.pca_audio_text_embeddings(
    text_embeddings=text_embeddings,
    audio_embeddings=audio_embeddings,
    n=n
)

# 2. Compute cosine similarity between audio embeddings and text embeddings
# a. Local embeddings
logger.info("Computing cosine similarity for local embeddings")
text_embeddings, audio_embeddings = evaluator.load_local_embeddings_test_set(n=n)

evaluator.audio_text_similarity(
    audio_embeddings=audio_embeddings,
    text_embeddings=text_embeddings,
    type_data="local"
)

# b. Global embeddings (avg over chunks per version_id)
logger.info("Computing cosine similarity for global embeddings")
df_version_ids_to_keep = evaluator.get_version_ids_to_keep()
text_embeddings, audio_embeddings = evaluator.load_local_embeddings_test_set_and_aggregate(df_version_ids_to_keep)
evaluator.audio_text_similarity(
    audio_embeddings=audio_embeddings,
    text_embeddings=text_embeddings,
    type_data="global"
)
"""