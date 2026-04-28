from pathlib import Path

import numpy as np
import pandas as pd
import ast

from .eval import run_optimization_experiments

# Default directory for optimization CSVs when ``output_dir`` is not passed to
# ``run_router_optimization_experiments``.
SOLVER_OUTPUT_DIR = Path(__file__).resolve().parent



def run_router_optimization_experiments(
    routers,
    gpu_counts,
    max_cost_per_query_list,
    batch_size,
    create_model_configuration,
    train_ratio=0.1,
    seed=42,
    verbose=True,
    output_dir=None,
):
    """
    Run optimization experiments for each router configuration.

    Writes ``optimization_results_{batch_size}_{router}_{prediction_label}_{test_name}_concurrency.csv``
    under ``output_dir`` if given, otherwise under ``solver/`` (same directory as this module).

    Parameters
    ----------
    routers : list of tuples
        Each tuple is (router_name, prediction_label, test_name, routing_csv_path)
    gpu_counts : list
        List of GPU budget values to test
    max_cost_per_query_list : list
        List of maximum cost per query values to test
    batch_size : int
        Batch size for optimization
    train_ratio : float, optional
        Fraction of questions for training (default: 0.1)
    seed : int, optional
        Random seed for train/test split (default: 42)
    verbose : bool, optional
        Whether to print progress (default: True)
    output_dir : str or pathlib.Path, optional
        Directory for optimization CSV output. Defaults to ``solver/`` (this module's parent).

    Returns
    -------
    list of pd.DataFrame
        Results DataFrames for each router
    """

    out_dir = Path(output_dir).resolve() if output_dir is not None else SOLVER_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for idx in range(len(routers)):
        router = routers[idx][0]
        prediction_label = routers[idx][1]
        test_name = routers[idx][2]
        routing_df = pd.read_csv(routers[idx][3])

        def _quantile_from_bootstrap(preds, q):
            try:
                if preds is None or (isinstance(preds, float) and np.isnan(preds)):
                    return np.nan
                arr = np.array(ast.literal_eval(preds) if isinstance(preds, str) else preds)
                return np.percentile(arr, q)
            except (ValueError, TypeError, SyntaxError):
                return np.nan

        if prediction_label == 'bootstrap_quantile_10':
            if prediction_label not in routing_df.columns:
                if 'bootstrap_predictions' not in routing_df.columns:
                    raise ValueError(
                        f"CSV must have 'bootstrap_predictions' to compute 'bootstrap_quantile_10'. "
                        f"Available columns: {list(routing_df.columns)}"
                    )
                routing_df[prediction_label] = routing_df["bootstrap_predictions"].apply(
                    lambda preds: _quantile_from_bootstrap(preds, 10)
                )
        elif prediction_label == 'bootstrap_quantile_20':
            if prediction_label not in routing_df.columns:
                if 'bootstrap_predictions' not in routing_df.columns:
                    raise ValueError(
                        f"CSV must have 'bootstrap_predictions' to compute 'bootstrap_quantile_20'. "
                        f"Available columns: {list(routing_df.columns)}"
                    )
                routing_df[prediction_label] = routing_df["bootstrap_predictions"].apply(
                    lambda preds: _quantile_from_bootstrap(preds, 20)
                )

        if prediction_label not in routing_df.columns:
            raise ValueError(
                f"Prediction column '{prediction_label}' not found in routing CSV. "
                f"Available columns: {list(routing_df.columns)}"
            )

        if verbose:
            print(router, prediction_label, routing_df.shape)
            print("Original routing_df shape:", routing_df.shape)

        # Split by query (train_ratio train, rest test)
        unique_questions = routing_df['question'].unique()
        n_questions = len(unique_questions)
        if verbose:
            print(f"Total unique questions: {n_questions}")

        np.random.seed(seed)
        shuffled_questions = np.random.permutation(unique_questions)
        n_train = int(train_ratio * n_questions)
        train_questions = shuffled_questions[:n_train]
        test_questions = shuffled_questions[n_train:]

        if verbose:
            print(f"Train questions: {len(train_questions)} ({len(train_questions)/n_questions*100:.1f}%)")
            print(f"Test questions: {len(test_questions)} ({len(test_questions)/n_questions*100:.1f}%)")

        routing_df_train = routing_df[routing_df['question'].isin(train_questions)].copy()
        routing_df_test = routing_df[routing_df['question'].isin(test_questions)].copy()

        if verbose:
            print(f"\nRouting df train shape: {routing_df_train.shape}")
            print(f"Routing df test shape: {routing_df_test.shape}")

        assert len(set(train_questions) & set(test_questions)) == 0, "Train and test questions overlap!"
        if verbose:
            print("✓ No overlap between train and test questions")

        routing_matrix = routing_df.pivot(
            index='question',
            columns='llm_name',
            values=prediction_label
        )
        model_names = routing_matrix.columns[1:].tolist() if 'question' in routing_matrix.columns else routing_matrix.columns.tolist()

        config_dict = create_model_configuration(model_names, verbose=verbose)

        g = config_dict['g']
        c = config_dict['c']
        l = config_dict['l']
        closed_source_models = config_dict['closed_source_models']

        results = run_optimization_experiments(
            routing_df_train=routing_df_train,
            routing_df_test=routing_df_test,
            gpu_counts=gpu_counts,
            max_cost_per_query_list=max_cost_per_query_list,
            c=c,
            g=g,
            l=l,
            df_model_names=model_names,
            closed_source_models=closed_source_models,
            batch_size=batch_size,
            prediction_label=prediction_label
        )

        rows = []
        for cost_val, metrics in results.items():
            for gpu_idx, gpu_budget in enumerate(gpu_counts):
                row = {
                    'cost_budget': cost_val,
                    'gpu_budget': gpu_budget,
                    'predicted_performance_train': metrics['performances'][gpu_idx],
                    'real_performance_train': metrics['real_performances'][gpu_idx],
                    'test_performance': metrics['test_performances'][gpu_idx],
                    'proportion_zero_actual': metrics['proportion_zero_actual'][gpu_idx],
                    'avg_cost_per_query': metrics['costs'][gpu_idx],
                    'gpus_used': metrics['gpus_used'][gpu_idx],
                    'small_instances': metrics['small_instances'][gpu_idx],
                    'large_instances': metrics['large_instances'][gpu_idx],
                    'opensource_instances': metrics['opensource_instances'][gpu_idx],
                    'closedsource_instances': metrics['closedsource_instances'][gpu_idx],
                    'prediction_label': prediction_label,
                    'router': router
                }
                rows.append(row)

        results_df = pd.DataFrame(rows)
        results_df = results_df.sort_values(['cost_budget', 'gpu_budget']).reset_index(drop=True)

        output_filename = f"optimization_results_{batch_size}_{router}_{prediction_label}_{test_name}_concurrency.csv"
        output_path = out_dir / output_filename

        results_df.to_csv(output_path, index=False)

        if verbose:
            print(f"Results saved to: {output_path}")
            print(f"Shape: {results_df.shape}")
            print(f"\nFirst few rows:")
            print(results_df.head(10))

        all_results.append(results_df)
    return all_results


def get_baseline_results(
    routing_csv_path,
    baseline_csv_path,
    gpu_counts,
    batch_size,
    create_model_configuration,
    evaluate_single_llm_baseline,
    train_ratio=0.05,
    seed=42,
    verbose=True,
):
    """
    Load routing data, split by query, and evaluate single-LLM baselines for all models.

    Parameters
    ----------
    routing_csv_path : str
        Path to the routing CSV file
    gpu_counts : list
        List of GPU budget values for baseline evaluation
    batch_size : int
        Batch size for evaluation
    create_model_configuration : callable
        Function to create model configuration (g, c, l, closed_source_models)
    evaluate_single_llm_baseline : callable
        Function to evaluate a single-LLM baseline
    train_ratio : float, optional
        Fraction of questions for train split (default: 0.05)
    seed : int, optional
        Random seed for train/test split (default: 42)
    verbose : bool, optional
        Whether to print progress (default: True)

    Returns
    -------
    dict
        Dictionary with keys: baseline_dict, routing_df_train, routing_df_test,
        model_names, model_configuration (g, c, l, closed_source_models)
    """
    # Accept str or single-element tuple (e.g. from trailing comma in notebook)
    if isinstance(routing_csv_path, tuple):
        routing_csv_path = routing_csv_path[0]
    if isinstance(baseline_csv_path, tuple):
        baseline_csv_path = baseline_csv_path[0]
    routing_df = pd.read_csv(routing_csv_path)

    if verbose:
        print("Original routing_df shape:", routing_df.shape)

    unique_questions = routing_df['question'].unique()
    n_questions = len(unique_questions)
    if verbose:
        print(f"Total unique questions: {n_questions}")

    np.random.seed(seed)
    shuffled_questions = np.random.permutation(unique_questions)
    n_train = int(train_ratio * n_questions)
    train_questions = shuffled_questions[:n_train]
    test_questions = shuffled_questions[n_train:]

    if verbose:
        print(f"Train questions: {len(train_questions)} ({len(train_questions)/n_questions*100:.1f}%)")
        print(f"Test questions: {len(test_questions)} ({len(test_questions)/n_questions*100:.1f}%)")

    routing_df_train = routing_df[routing_df['question'].isin(train_questions)].copy()
    routing_df_test = routing_df[routing_df['question'].isin(test_questions)].copy()

    model_names = routing_df_train['llm_name'].unique().tolist()
    if verbose:
        print(f"\nFound {len(model_names)} unique models: {model_names}")

    if verbose:
        print(f"\nEvaluating baselines for {len(model_names)} models...")
        print("="*70)

    baseline_dict = {}
    model_configuration = create_model_configuration(model_names=model_names, verbose=False)
    g = model_configuration['g']
    c = model_configuration['c']
    l = model_configuration['l']
    closed_source_models = model_configuration['closed_source_models']

    for llm_name in model_names:
        try:
            if verbose:
                print(f"\nEvaluating baseline for {llm_name}...")
            baseline_results = evaluate_single_llm_baseline(
                llm_name=llm_name,
                gpu_counts=gpu_counts,
                df_model_names=model_names,
                routing_df_test=routing_df_test,
                c=c,
                g=g,
                l=l,
                closed_source_models=closed_source_models,
                batch_size=batch_size,
            )

            cost_key = list(baseline_results.keys())[0]
            test_perfs = baseline_results[cost_key]['test_performances']
            test_perf = next((p for p in test_perfs if p is not None), None)

            if test_perf is not None:
                baseline_dict[llm_name] = baseline_results
                if verbose:
                    print(f"✓ Completed baseline for {llm_name} (test perf: {test_perf:.4f})")
        except Exception as e:
            if verbose:
                print(f"✗ Failed baseline for {llm_name}: {e}")
    
    # Save baseline_dict as a dataframe to CSV
    baseline_columns = [
        'llm_name', 'gpu_budget', 'test_performance', 'cost', 'gpus_used',
        'small_instances', 'large_instances', 'opensource_instances',
        'closedsource_instances', 'predicted_performance', 'real_performance',
    ]
    rows = []
    for llm_name, baseline_results in baseline_dict.items():
        cost_key = list(baseline_results.keys())[0]
        metrics = baseline_results[cost_key]
        for gpu_idx, gpu_budget in enumerate(gpu_counts):
            rows.append({
                'llm_name': llm_name,
                'gpu_budget': gpu_budget,
                'test_performance': metrics['test_performances'][gpu_idx],
                'cost': metrics['costs'][gpu_idx],
                'gpus_used': metrics['gpus_used'][gpu_idx],
                'small_instances': metrics['small_instances'][gpu_idx],
                'large_instances': metrics['large_instances'][gpu_idx],
                'opensource_instances': metrics['opensource_instances'][gpu_idx],
                'closedsource_instances': metrics['closedsource_instances'][gpu_idx],
                'predicted_performance': metrics['performances'][gpu_idx],
                'real_performance': metrics['real_performances'][gpu_idx],
            })
    baseline_df = pd.DataFrame(rows, columns=baseline_columns if not rows else None)
    baseline_path = Path(baseline_csv_path)
    if not baseline_path.is_absolute():
        baseline_path = SOLVER_OUTPUT_DIR / baseline_path.name
    baseline_df.to_csv(baseline_path, index=False)
    if not rows:
        print(f"WARNING: No baseline results (all models failed). Wrote empty CSV with headers to {baseline_path}")
    else:
        print(f"Baseline results saved to {baseline_path}")

    return {
        'baseline_dict': baseline_dict,
        'routing_df_train': routing_df_train,
        'routing_df_test': routing_df_test,
        'model_names': model_names,
        'model_configuration': model_configuration,
    }
