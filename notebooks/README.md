# Notebooks

Run from **`llm_routing/notebooks/`** (or ensure the first code cell can walk up to `llm_routing/`).

## Re-executing and refreshing outputs

Install Jupyter tooling (e.g. `pip install nbconvert ipykernel`).

**`batch_routing.ipynb`** — the batch optimization and baseline cells can take a long time. For a faster pass that only refreshes paths, skip messages, and **plots** (using existing `optimization_results_*.csv` under `results/routing_results/`):

```bash
cd notebooks
NB_SKIP_LONG_BATCH=1 MPLBACKEND=Agg python3 -m nbconvert --to notebook --execute batch_routing.ipynb --inplace
```

Unset `NB_SKIP_LONG_BATCH` (or set it to `0`) to re-run full CVXPY optimization and baselines.

**`per_query_routing.ipynb`** — the per-query sweep over routers can take a long time. To skip sweeps and only run downstream cells that use existing `results_per_query_*.csv`:

```bash
cd notebooks
NB_SKIP_LONG_PER_QUERY=1 MPLBACKEND=Agg python3 -m nbconvert --to notebook --execute per_query_routing.ipynb --inplace
```

Environment variables are documented in the first markdown cell of each notebook.
