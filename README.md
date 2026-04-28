# LLM Routing Pipeline

Pipeline for **LLM performance estimation** and **routing** (per-query cost/performance sweeps and batch GPU/cost optimization). Training and evaluation use **Dataset 1** from Song et al. (2025) (IRT-Router), stored under `data/irt_data/` and tracked with **Git LFS** (see [`data/README.md`](data/README.md)).

## Quick start

```bash
cd llm_routing
git lfs install
git lfs pull   # materialize Dataset 1 CSVs if clone used LFS pointers only
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run all CLI examples **from this directory** (`llm_routing/`) so imports and paths resolve.

## What is committed

- **`data/irt_data/*.csv`** — Dataset 1, via **Git LFS** (install [Git LFS](https://git-lfs.com) before clone/push).
- **Tracked in plain Git:** Python sources, `notebooks/`, `utils/map/*.csv`, `data/README.md`.

**Gitignored** (large or regenerable; not in version control):

- `utils/bert_embeddings/*.pkl`, `utils/cold/*.pkl`, `utils/relevance/*.pkl` — embeddings / auxiliary pickles used by training and `test_models`
- `results/` — trained snapshots, performance-estimate CSVs, routing outputs (reproduced by the steps below)

## Repository structure

```
llm_routing/
├── data/
│   ├── README.md           # Dataset 1 (Git LFS)
│   └── irt_data/           # train.csv, test1.csv, test2.csv (Git LFS)
├── train/                  # Training scripts (MIRT, XGBoost)
├── test/                   # Performance estimates on test splits
├── routing/                # Per-query + batch routing
│   └── solver/             # CVXPY batch optimizer
├── utils/                  # ID maps; large pickles gitignored (see above)
├── notebooks/              # batch_routing, per_query_routing, plotting
├── results/                # Generated (gitignored)
└── requirements.txt
```

## Step 1: Model performance estimates

### A. Train the performance model

**XGBoost with bootstrap:**

```bash
python3 train/train_xgboost.py --embedding bert --bootstrap 100
```

**MIRT (no bootstrap):**

```bash
python3 train/train_mirt.py --embedding bert
```

Outputs: snapshots under `results/trained_models/` (gitignored).

### B. Compute performance estimates on test data

**MIRT:**

```bash
python3 -m test.test_models --router mirt --emb_name bert --test_path test1 --a 0.8
```

**XGBoost (bootstrap):**

```bash
python3 -m test.test_models --router xgboost --emb_name bert --test_path test1 --n_bootstrap 100
```

**k-NN (bootstrap):**

```bash
python3 -m test.test_models --router knn --emb_name bert --test_path test1 --k_neighbors 40 --n_bootstrap 100
```

Outputs: CSVs under `results/performance_estimates/` (gitignored).

## Step 2: Query routing

### C. Per-query routing

```bash
python3 routing/per_query_routing.py
```

Outputs: CSVs (and optional plots) under `results/routing_results/`.

### D. Batch routing

```bash
python3 routing/batch_optimization.py
```

Writes optimization and baseline CSVs under `results/routing_results/` (see `routing/batch_optimization.py`).

## Notebooks

Prefer running Jupyter with **cwd = `llm_routing/notebooks/`** (or `llm_routing/`) so path setup cells resolve; notebooks are checked in **without stored outputs**.

## Parent checkout (`query-routing`)

This tree can live inside a larger monorepo. It has its **own `.git`** directory; treat it as a separate repository (submodule or standalone clone), not as ordinary nested files of the parent Git project.
