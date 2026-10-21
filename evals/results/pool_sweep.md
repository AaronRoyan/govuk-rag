# Reranker candidate pool sweep

| Mode | Pool | Hit@1 | Hit@3 | MRR | p50 ms |
|---|---|---|---|---|---|
| `dense_rerank` | 10 | 0.800 | 0.971 | 0.881 | 179.3 |
| `dense_rerank` | 20 | 0.800 | 0.971 | 0.880 | 381.0 |
| `dense_rerank` | 30 | 0.814 | 0.986 | 0.897 | 607.2 |
| `dense_rerank` | 50 | 0.814 | 0.971 | 0.893 | 1124.4 |
| `hybrid_rerank` | 10 | 0.814 | 0.986 | 0.895 | 198.8 |
| `hybrid_rerank` | 20 | 0.814 | 0.986 | 0.895 | 412.4 |
| `hybrid_rerank` | 30 | 0.814 | 0.986 | 0.894 | 635.8 |
| `hybrid_rerank` | 50 | 0.814 | 0.986 | 0.893 | 1159.1 |
