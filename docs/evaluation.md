# Evaluation

This project executes a reproducible ablation study comparing baseline and proposed variants.

## Metrics

The evaluation module supports:

- Precision@K
- Recall@K
- F1
- MRR
- NDCG
- contradiction accuracy and related metrics

These metrics are only computed when valid labels or retrieval orderings are available. The implementation does not fabricate missing measurement results.

## Log outputs

The experimental results are saved under results/ and can be used for tables and plots.
