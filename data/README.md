# Data directory

**Dataset 1** (train/test splits used in Song et al., IRT-Router) is **not** stored in this repository. Download the same files from the upstream project:

**[IRT-Router `data/`](https://github.com/Mercidaiha/IRT-Router/tree/main/data)** — you need `train.csv`, `test1.csv`, and `test2.csv`.

Place them under this repo as:

- `irt_data/train.csv`
- `irt_data/test1.csv`
- `irt_data/test2.csv`

### Example (command line)

From the `llm_routing/` root:

```bash
mkdir -p data/irt_data
BASE="https://raw.githubusercontent.com/Mercidaiha/IRT-Router/main/data"
for f in train.csv test1.csv test2.csv; do
  curl -fL -o "data/irt_data/$f" "$BASE/$f"
done
```

Then run the training and evaluation steps in the root [`README.md`](../README.md).
