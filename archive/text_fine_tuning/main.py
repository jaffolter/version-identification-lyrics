import pandas as pd 
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from datasets import load_dataset
import torch, random
from itertools import combinations

# 1. Remove strange examples
df = pd.read_csv("src/data/discogs/discogs_vi_final.csv")
print(df)

""" 
res = df.groupby('clique_id').size().reset_index(name='count')
print(res)


print(res[res['count'] > 1].sort_values(by='count', ascending=False))


final_df = df[df['clique_id'].isin(res[res['count'] > 1]['clique_id'])].reset_index(drop=True)
print(final_df.head())
print(len(final_df))

final_df.to_csv("src/data/discogs/discogs_vi_final_filtered.csv", index=False)



model_name= "Alibaba-NLP/gte-multilingual-base"
model = SentenceTransformer(model_name, trust_remote_code=True) 
print(df.columns)
bad_md5 = []
for clique_id, group in tqdm(df.groupby('clique_id'), total=df['clique_id'].nunique()):
    if len(group) < 2:
        continue  
    lyrics = group['transcription_vocal'].tolist()
    ids = group['md5_encoded'].tolist()
    
    embeddings = model.encode(lyrics, normalize_embeddings=True, device='cuda')
    
    # Compute pairwise cosine similarity
    cosine_similarities = embeddings @ embeddings.T
    #print(cosine_similarities)
    
    # Compute mean over each row 
    np.fill_diagonal(cosine_similarities, 0)

    # Since the diagonal is excluded, divide by (n - 1) instead of n
    n = cosine_similarities.shape[1]
    mean_similarities = cosine_similarities.sum(axis=1) / (n - 1)
    #print(mean_similarities)
    # If mean < 0.5, we consider it a bad clique
    for idx, id in enumerate(ids):
        if mean_similarities[idx] < 0.7:
            print(f"Bad clique {clique_id} with id {id} and mean similarity {mean_similarities[idx]}")
            bad_md5.append(id)
            

bad_md5 = list(set(bad_md5))
print(f"Total bad md5: {len(bad_md5)}")

good_df = df[~df['md5_encoded'].isin(bad_md5)]
print(good_df.head())
print(good_df.head())
print(len(good_df))

good_df.to_csv("src/data/discogs/good_cliques.csv", index=False)



# 2. Mine hard negatives from good cliques

# Load your data
print("Loading CSV...")
df = pd.read_csv("src/data/discogs/good_cliques.csv")
print(f"Loaded {len(df)} rows from good_cliques.csv")

model_name = "Alibaba-NLP/gte-multilingual-base"
print(f"Loading model: {model_name}")
model = SentenceTransformer(model_name, trust_remote_code=True)

# Compute embeddings (or load precomputed ones)
print("Encoding transcriptions...")
transcriptions = df['transcription_vocal'].fillna("").tolist()

embeddings = model.encode(transcriptions, normalize_embeddings=True, device='cuda', show_progress_bar=True, batch_size=32)
# Save them 
print("Saving embeddings...")
np.savez_compressed("src/data/discogs/good_cliques_embeddings.npz", embeddings=embeddings)
"""

# Load embeddings
df = pd.read_csv("src/data/discogs/good_cliques.csv")

embeddings = np.load("src/data/discogs/good_cliques_embeddings.npz")['embeddings']

# Filter df so it only contains the embeddings


df['embedding'] = [torch.tensor(emb).cpu().numpy() for emb in embeddings]

# Prepare list of embeddings per sample
print("Preparing structures...")
embeddings = np.stack(df['embedding'].values)
clique_ids = df['clique_id'].values
md5_ids = df['md5_encoded'].values

# For fast lookup
md5_to_index = {md5: i for i, md5 in enumerate(md5_ids)}
clique_to_indices = df.groupby('clique_id').indices


# Step 2: Build hard negatives
print("Mining hard negatives...")
csv_path = "src/data/discogs/hard_negatives.csv"
pd.DataFrame(columns=["anchor", "positive", "hard_negative", "sim_ap", "sim_an"]).to_csv(csv_path, index=False)

for clique_id, indices in tqdm(clique_to_indices.items(), desc="Processing cliques"):
    hard_negatives = []

    if len(indices) < 2:
        continue  # can't form anchor-positive pairs

    for anchor_idx, pos_idx in combinations(indices, 2):
        anchor_emb = embeddings[anchor_idx]
        pos_emb = embeddings[pos_idx]

        sim_ap = np.dot(anchor_emb, pos_emb)  # Cosine similarity
        if sim_ap < 0.8:
            print(f"Skipping low similarity pair: {md5_ids[anchor_idx]} vs {md5_ids[pos_idx]} (sim_ap={sim_ap:.3f})")
            continue

        # Mask out positives (same clique)
        neg_mask = np.ones(len(df), dtype=bool)
        neg_mask[indices] = False

        # Get embeddings and IDs for all negatives
        neg_embs = embeddings[neg_mask]
        neg_ids = md5_ids[neg_mask]

        # Compute similarities between anchor and all negatives
        sim_an = neg_embs @ anchor_emb  # (N_negatives,)

        # Find hard negatives
        hard_idxs = np.where(sim_an > sim_ap)[0]
        if len(hard_idxs) > 0:
            print(
                f"[clique {clique_id}] Anchor {md5_ids[anchor_idx]} vs Pos {md5_ids[pos_idx]} "
                f"(sim_ap={sim_ap:.3f}) ➝ {len(hard_idxs)} hard negatives found."
            )

        for idx in hard_idxs:
            hard_neg_md5 = neg_ids[idx]
            sim_an_val = sim_an[idx]
            hard_negatives.append({
                'anchor': md5_ids[anchor_idx],
                'positive': md5_ids[pos_idx],
                'hard_negative': hard_neg_md5,
                'sim_ap': sim_ap,
                'sim_an': sim_an_val
            })

        batch_df = pd.DataFrame(hard_negatives)
        batch_df.to_csv(csv_path, mode='a', header=False, index=False)

            

# Convert to DataFrame
#hard_negatives_df = pd.DataFrame(hard_negatives)
#print(f"Total hard negatives mined: {len(hard_negatives_df)}")
#print(hard_negatives_df.head())

# Save to CSV
#output_path = "src/data/discogs/hard_negatives.csv"
#hard_negatives_df.to_csv(output_path, index=False)
#print(f"Hard negatives saved to: {output_path}")


# 3. Train 

print("Starting training...")


# 1. Load model
model = SentenceTransformer(
    "Alibaba-NLP/gte-multilingual-base",
    trust_remote_code=True,
)

# 2. Load triplet CSV
raw_ds = load_dataset("csv", data_files="src/data/discogs/hard_negatives.csv", split="train")

# Convert to InputExample
def to_input_example(row):
    return InputExample(
        texts=[row["anchor"], row["positive"], row["hard_negative"]],
        label=1.0
    )

examples = [to_input_example(r) for r in raw_ds]

# Shuffle and split
random.seed(42)
random.shuffle(examples)
split_idx = int(0.8 * len(examples))
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]

# 3. Create evaluator from validation triplets
anchor_texts = [ex.texts[0] for ex in val_examples]
positive_texts = [ex.texts[1] for ex in val_examples]
negative_texts = [ex.texts[2] for ex in val_examples]

evaluator = TripletEvaluator(
    anchor_sentences=anchor_texts,
    positive_sentences=positive_texts,
    negative_sentences=negative_texts,
    name="triplet-eval",
    batch_size=128,
    main_similarity="cosine"
)

# 4. Training args
args = SentenceTransformerTrainingArguments(
    output_dir="src/checkpoints/alibaba_triplet",
    overwrite_output_dir=False,
    num_train_epochs=4,
    per_device_train_batch_size=128,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    gradient_accumulation_steps=4,
    bf16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=100,
    save_total_limit=3,
    run_name="version_id_triplet",
    seed=42,
)

# 5. Trainer
loss_fct = losses.TripletLoss(
    model=model,
    distance_metric=losses.TripletDistanceMetric.COSINE,
    margin=0.3
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_examples,
    loss_fct=loss_fct,
    tokenizer=model.tokenizer,
    evaluator=evaluator  # ← this is the correct place to add it!
)

trainer.train()
