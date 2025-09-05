import torch


def retrieval_metrics(
    a: torch.Tensor, b: torch.Tensor, topk: list[int] = [1, 5, 10], map_k: int = 10
) -> dict[str, float]:
    """
    Compute retrieval metrics between two aligned embedding sets.

    Designed for cross-modal retrieval (e.g., audio ↔ lyrics), where
    each row in `a` corresponds to its matching row in `b`.

    Metrics:
        - HR@k (Hit Rate @ k): proportion of queries where the true match
          is ranked within the top-k retrieved items.
        - MAP@k (Mean Average Precision @ k): average reciprocal rank of the
          true match, truncated at k. Equivalent to MRR when there is only
          one relevant item per query.
        - Cosine similarity (diag): mean cosine similarity between matched pairs.

    Args:
        a (torch.Tensor): Embeddings of shape (B, D) for modality A (queries).
        b (torch.Tensor): Embeddings of shape (B, D) for modality B (candidates).
        topk (list[int], optional): List of cutoff values for HR@k. Default: [1, 5, 10].
        map_k (int, optional): Cutoff for MAP@k. Default: 10.

    Returns:
        dict[str, float]: Retrieval metrics { "HR@k": float, "MAP@k": float, "cosine_sim": float }
    """
    # Cosine similarity matrix (B x B) since a and b are L2-normalized embeddings
    sim = a @ b.T
    batch_size = sim.size(0)
    targets = torch.arange(batch_size, device=a.device)

    metrics: dict[str, float] = {}
    max_k = min(batch_size, max(max(topk), map_k))  # ensure k ≤ batch size

    # --- Hit Rate @ k ---
    # For each query, get indices of top-k candidates
    _, top_idx = sim.topk(k=max_k, dim=-1)
    for k in topk:
        # Check if the true match is among the top-k
        hits = (top_idx[:, :k] == targets[:, None]).any(dim=-1).float().mean()
        metrics[f"HR@{k}"] = hits.detach().cpu().item()

    # --- MAP @ k ---
    # Since there is only one relevant item per query, MAP@k reduces to
    # reciprocal rank of the true match (if within top-k), else 0
    hits_at_k = top_idx[:, :map_k] == targets[:, None]
    # First occurrence (rank) of the correct item
    pos = torch.where(
        hits_at_k.any(dim=1),
        hits_at_k.float().argmax(dim=1) + 1,  # +1 → ranks are 1-based
        torch.full_like(targets, map_k + 1),  # not found → assign rank > k
    )
    ap = torch.where(pos <= map_k, 1.0 / pos.float(), torch.zeros_like(pos, dtype=torch.float))
    metrics[f"MAP@{map_k}"] = ap.mean().detach().cpu().item()

    # --- Cosine similarity of ground-truth pairs ---
    metrics["cosine_sim"] = sim.diag().mean().detach().cpu().item()

    return metrics
