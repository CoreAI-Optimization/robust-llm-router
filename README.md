# Roubust LLM Routing

Pipeline for **LLM performance estimation** and **routing** (per-query cost/performance sweeps and batch GPU/cost optimization). Training and evaluation use dataset from Song et al. (2025) (IRT-Router). Download train and test data from **[Mercidaiha/IRT-Router](https://github.com/Mercidaiha/IRT-Router/tree/main/data)** `data/` into `data/irt_data/` (see `[data/README.md](data/README.md)`).

## Quick start

```bash
cd llm_routing
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download **dataset** into `data/irt_data/` (see `[data/README.md](data/README.md)` for `curl` one-liners from the IRT-Router repo).

Download **utils** artifacts (embeddings, cold-start, relevance, maps) from **[IRT-Router `utils/](https://github.com/Mercidaiha/IRT-Router/tree/main/utils)`** — see `[utils/README.md](utils/README.md)` for `curl` commands and folder layout.

Run all CLI examples **from this directory** (`llm_routing/`) so imports and paths resolve.

## Repository structure

```
llm_routing/
├── data/
│   ├── README.md           # how to fetch Dataset 1 from IRT-Router
│   └── irt_data/           # train.csv, test1.csv, test2.csv (local; gitignored)
├── train/                  # Training scripts (MIRT, XGBoost)
├── test/                   # Performance estimates on test splits
├── routing/                # Per-query + batch routing
│   └── solver/             # CVXPY batch optimizer
├── utils/                  # embeddings, maps, etc. (see utils/README.md; pkls from IRT-Router)
├── notebooks/              # batch_routing, per_query_routing, plotting
├── results/                # Tracked via Git LFS (snapshots, PDFs, large CSVs)
│   ├── trained_models/     # MIRT and XGBoost snapshots
│   ├── performance_estimates/  # Per-query performance estimate CSVs (MIRT, XGBoost, k-NN)
│   └── routing_results/    # Per-query and batch routing outputs (CSVs, PDFs)
└── requirements.txt
```

> **Large files:** `results/trained_models/*.snapshot`, `results/performance_estimates/*.csv`, and `results/routing_results/*.pdf` are stored with [Git LFS](https://git-lfs.github.com). Run `git lfs pull` after cloning to download them.

## Step 1: Model performance estimates

Before training or running `test_models`:

1. Download `**train.csv**`, `**test1.csv**`, and `**test2.csv**` from [IRT-Router `data/](https://github.com/Mercidaiha/IRT-Router/tree/main/data)` into `data/irt_data/` (commands in `[data/README.md](data/README.md)`).
2. Download **utils** (embeddings, cold, relevance, maps as needed) from [IRT-Router `utils/](https://github.com/Mercidaiha/IRT-Router/tree/main/utils)` (commands in `[utils/README.md](utils/README.md)`).

### A. Train the performance model

**XGBoost with bootstrap:**

```bash
python3 train/train_xgboost.py --embedding bert --bootstrap 100
```

**MIRT (no bootstrap):**

```bash
python3 train/train_mirt.py --embedding bert
```

Outputs: snapshots under `results/trained_models/` (tracked via Git LFS).

### B. Compute performance estimates on test data

**MIRT:**

```bash
python3 -m test.test_models --router mirt --emb_name bert --test_path test1
```

**XGBoost (bootstrap):**

```bash
python3 -m test.test_models --router xgboost --emb_name bert --test_path test1 --n_bootstrap 100
```

**k-NN (bootstrap):**

```bash
python3 -m test.test_models --router knn --emb_name bert --test_path test1 --k_neighbors 40 --n_bootstrap 100
```

Outputs: CSVs under `results/performance_estimates/` (tracked via Git LFS).

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

Prefer running Jupyter with **cwd = `llm_routing/notebooks/`** (or `llm_routing/`) so path setup cells resolve. Notebooks are checked in **with executed outputs** (text and tables; figures may also be saved as PDFs under `results/routing_results/`).

To **re-run** them non-interactively (optionally skipping long optimization/sweep cells while still refreshing plots), see `[notebooks/README.md](notebooks/README.md)`.

