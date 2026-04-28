import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # llm_routing/ on path

import ast
import pandas as pd
import numpy as np

from test.test_models import evaluate_routing

LLM_ROUTING_DIR = Path(__file__).parent.parent
PERF_EST_DIR = LLM_ROUTING_DIR / "results" / "performance_estimates"
RESULTS_DIR = LLM_ROUTING_DIR / "results" / "routing_results"


def sweep_performance_cost(
    routing_df,
    router=None,
    emb_name='bert',
    test_path='test1',
    prediction_labels=None,
    a_values=None,
    output_csv=None,
):
    """
    Sweep over a values (performance weight) for different prediction labels.

    Args:
        routing_df: Precomputed performance-estimates DataFrame
            (from compute_mirt/xgboost/knn_performance_estimates).
        router: Router label used only for logging.
        emb_name: Embedding type, used for logging (e.g. 'bert').
        test_path: Test split name passed to evaluate_routing (e.g. 'test1').
        prediction_labels: Columns in routing_df to use as the performance signal.
            Default: ['main_model_prediction', 'bootstrap_quantile_2_5']
        a_values: Performance-weight values to sweep (a=1 → pure performance).
            Default: np.linspace(0, 1, 11)
        output_csv: If given, save a summary CSV to this path.

    Returns:
        dict: {label: {'a_values': [...], 'performance': [...], 'cost': [...]}}
    """
    if prediction_labels is None:
        prediction_labels = ['main_model_prediction', 'bootstrap_quantile_2_5']
    if a_values is None:
        a_values = np.linspace(0.0, 1.0, 11)

    results = {label: {'a_values': [], 'performance': [], 'cost': []} for label in prediction_labels}

    print(f"Router: {router}  |  test_path: {test_path}  |  emb: {emb_name}")
    print(f"Records: {len(routing_df)}  |  unique questions: {len(routing_df['question'].unique())}")
    print(f"Columns: {routing_df.columns.tolist()}")

    for label in prediction_labels:
        print(f"\n{'='*60}\nPrediction label: {label}\n{'='*60}")

        for a in a_values:
            try:
                avg_perf, total_cost, opt_cost, chosen_llms = evaluate_routing(
                    routing_df=routing_df.copy(),
                    test_path=test_path,
                    a=a,
                    task=None,
                    candidate_llms=None,
                    prediction_label=label,
                )
                results[label]['a_values'].append(a)
                results[label]['performance'].append(avg_perf)
                results[label]['cost'].append(total_cost)
                print(f"  a={a:.2f}  perf={avg_perf:.4f}  cost={total_cost:.6f}")

            except Exception as e:
                import traceback
                print(f"  ERROR at a={a:.2f}: {e}")
                traceback.print_exc()

    if output_csv:
        rows = [
            {'prediction_label': label, 'a_value': a, 'performance': p, 'cost': c}
            for label, data in results.items()
            for a, p, c in zip(data['a_values'], data['performance'], data['cost'])
        ]
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"\nSaved: {output_csv}")

    return results


def _compute_derived_label(routing_df, prediction_label):
    """Add derived prediction columns to routing_df in-place."""
    if prediction_label == 'bootstrap_quantile_10':
        routing_df["bootstrap_quantile_10"] = routing_df["bootstrap_predictions"].apply(
            lambda p: np.percentile(np.array(ast.literal_eval(p) if isinstance(p, str) else p), 10)
        )


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_name = "test1"
    np.random.seed(1)

    bootstrap_labels = ['bootstrap_quantile_10', 'performance_prediction']

    routers = [
        ('xgboost', PERF_EST_DIR / f'xgboost_bootstrap_100_bert_{test_name}.csv', bootstrap_labels),
        ('knn_5',   PERF_EST_DIR / f'knn_5_bootstrap_100_bert_{test_name}.csv',   bootstrap_labels),
        ('knn_40',  PERF_EST_DIR / f'knn_40_bootstrap_100_bert_{test_name}.csv',  bootstrap_labels),
        ('mirt',    PERF_EST_DIR / f'mirt_bert_{test_name}.csv',              ['performance_prediction']),
    ]

    for router_name, routing_df_csv, prediction_labels in routers:
        if not Path(routing_df_csv).exists():
            print(f"Skipping {router_name}: {routing_df_csv} not found")
            continue

        routing_df = pd.read_csv(routing_df_csv)

        for prediction_label in prediction_labels:
            _compute_derived_label(routing_df, prediction_label)

            sweep_performance_cost(
                routing_df=routing_df,
                router=None,
                emb_name='bert',
                test_path=test_name,
                prediction_labels=[prediction_label],
                a_values=np.linspace(0.0, 1.0, 11),
                output_csv=RESULTS_DIR / f'results_per_query_{router_name}_{prediction_label}_{test_name}.csv'
            )
