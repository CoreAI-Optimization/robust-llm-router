# Utils directory

Large embedding pickles and related artifacts used by `train/` and `test/test_models.py` are **not** committed to this repository. Get them from the upstream **[IRT-Router `utils/`](https://github.com/Mercidaiha/IRT-Router/tree/main/utils)** (same layout as here).

## Layout you need under `llm_routing/utils/`

| Path (under `utils/`) | Purpose |
|----------------------|---------|
| `bert_embeddings/llm_embeddings.pkl`, `query_embeddings.pkl` | BERT embeddings for LLMs and queries |
| `cold/test_avg_embeddings_bert.pkl` | Cold-start averages (BERT) |
| `relevance/relevance_vectors_cluster_train_bert.pkl`, `relevance_vectors_cluster_test_bert.pkl` | Relevance clustering vectors |
| `map/llm.csv`, `map/query.csv` | ID maps (also tracked in this repo; re-fetch from upstream if you want exact IRT-Router copies) |

## Option A — `curl` (BERT paths used by default scripts)

From the **`llm_routing/`** repository root:

```bash
BASE="https://raw.githubusercontent.com/Mercidaiha/IRT-Router/main/utils"

mkdir -p utils/bert_embeddings utils/cold utils/relevance utils/map

curl -fL -o utils/bert_embeddings/llm_embeddings.pkl       "$BASE/bert_embeddings/llm_embeddings.pkl"
curl -fL -o utils/bert_embeddings/query_embeddings.pkl       "$BASE/bert_embeddings/query_embeddings.pkl"
curl -fL -o utils/cold/test_avg_embeddings_bert.pkl          "$BASE/cold/test_avg_embeddings_bert.pkl"
curl -fL -o utils/relevance/relevance_vectors_cluster_train_bert.pkl  "$BASE/relevance/relevance_vectors_cluster_train_bert.pkl"
curl -fL -o utils/relevance/relevance_vectors_cluster_test_bert.pkl   "$BASE/relevance/relevance_vectors_cluster_test_bert.pkl"
curl -fL -o utils/map/llm.csv                                "$BASE/map/llm.csv"
curl -fL -o utils/map/query.csv                              "$BASE/map/query.csv"
```

For other embedding types (`open`, `zhipu`, `bge`), download the matching files from the same upstream tree (see folders under [`utils/`](https://github.com/Mercidaiha/IRT-Router/tree/main/utils)).

## Option B — Copy from a full IRT-Router clone

If you already cloned [Mercidaiha/IRT-Router](https://github.com/Mercidaiha/IRT-Router), from **`llm_routing/`**:

```bash
IRT=/path/to/IRT-Router
for d in bert_embeddings cold relevance map; do
  mkdir -p "utils/$d"
  rsync -a --exclude='__pycache__' --exclude='.DS_Store' "$IRT/utils/$d/" "utils/$d/"
done
```

Adjust `IRT` to your clone path.

Then continue with the root [`README.md`](../README.md).
