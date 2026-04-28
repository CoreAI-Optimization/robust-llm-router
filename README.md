# Robust LLM Routing

Pipeline for **LLM performance estimation** and **routing** (per-query cost/performance sweeps and batch GPU/cost optimization). Training and evaluation use the dataset from Song et al. (2025) (IRT-Router).

## Quick start

```bash
git clone https://github.com/CoreAI-Optimization/robust-llm-router.git
cd robust-llm-router
git lfs pull                  # download large files (dataset, embeddings, results)
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run all CLI examples **from this directory** (`llm_routing/`) so imports and paths resolve.

## Repository structure

```
llm_routing/
├── data/
│   └── irt_data/               # train.csv, test1.csv (Git LFS)
├── train/                      # Training scripts (MIRT, XGBoost)
├── test/                       # Performance estimates on test splits
├── routing/                    # Per-query + batch routing
│   └── solver/                 # CVXPY batch optimizer
├── utils/
│   ├── bert_embeddings/        # LLM and query embeddings (Git LFS)
│   ├── cold/                   # Cold-start embeddings (Git LFS)
│   ├── relevance/              # Relevance vectors (Git LFS)
│   └── map/                    # LLM and query index maps
├── notebooks/                  # batch_routing, per_query_routing, plotting
├── results/                    # Precomputed outputs (Git LFS)
│   ├── trained_models/         # MIRT and XGBoost snapshots
│   ├── performance_estimates/  # Per-query performance estimate CSVs
│   └── routing_results/        # Per-query and batch routing outputs (CSVs, PDFs)
└── requirements.txt
```

> **Large files:** dataset CSVs, embeddings (`.pkl`), trained model snapshots, and result CSVs/PDFs are all stored via [Git LFS](https://git-lfs.github.com). Run `git lfs pull` after cloning to download them.

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

Writes optimization and baseline CSVs under `results/routing_results/`.

## Notebooks

Prefer running Jupyter with **cwd = `llm_routing/notebooks/`** (or `llm_routing/`) so path setup cells resolve. Notebooks are checked in **with executed outputs** (text and tables; figures may also be saved as PDFs under `results/routing_results/`).

To **re-run** them non-interactively (optionally skipping long optimization/sweep cells while still refreshing plots), see [notebooks/README.md](notebooks/README.md).
