import numpy as np
import pandas as pd
import time
import sys
import os
import pickle
import subprocess
from collections import Counter
from .optimizer import (
    optimize_model_assignments_train,
    optimize_model_assignments_test,
    optimize_model_assignments_test_with_timeout,
    optimize_model_assignments_robust_test,
)

# Time limit per GLPK solver attempt (each mip_gap try). Up to 4 mip_gaps may be tried.
DEFAULT_MAX_OPTIMIZATION_TIME_PER_MIP_ATTEMPT = None


def run_cost_only_optimizer_with_timeout(a, c, C, timeout_seconds):
    """
    Run optimize_model_assignments_cost_only in a separate process with a hard time limit.
    Uses subprocess (not multiprocessing) so the OS enforces the timeout reliably,
    including when run from Jupyter. Returns (x_opt, timed_out).
    """
    timeout_seconds = int(max(1, timeout_seconds))
    _solver_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(_solver_dir)
    worker_module = "solver.cost_only_worker_main"
    try:
        result = subprocess.run(
            [sys.executable, "-m", worker_module],
            input=pickle.dumps((a, c, C)),
            capture_output=True,
            timeout=timeout_seconds,
            cwd=project_root,
        )
    except subprocess.TimeoutExpired:
        return None, True
    if result.returncode != 0:
        return None, True
    try:
        x_opt = pickle.loads(result.stdout)
    except Exception:
        return None, True
    return x_opt, False


def create_random_batches(n_queries, batch_size, shuffle=True):
    """
    Create random batches of queries.
    
    Parameters
    ----------
    n_queries : int
        Total number of queries
    batch_size : int
        Size of each batch
    shuffle : bool
        Whether to shuffle the queries before batching (default: True)
    
    Returns
    -------
    batches : list of lists
        List where each element is a list of query indices in that batch
    """
    indices = np.arange(n_queries)
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Split into batches
    batches = []
    for i in range(0, n_queries, batch_size):
        batch = indices[i:min(i + batch_size, n_queries)].tolist()
        batches.append(batch)
    
    return batches


def evaluate_on_test_set(routing_df_test=None, df_model_names=None, 
                         I_opt=None, c=None, l=None, batch_size=None, 
                         max_cost_per_query=None, optimize_func=None, 
                         prediction_label=None, n_test_batches=None, 
                         random_state=42, verbose=True,
                         max_optimization_time_per_batch=DEFAULT_MAX_OPTIMIZATION_TIME_PER_MIP_ATTEMPT,
                         timeout_bounds_csv_path="timeout_bounds.csv"):
    """
    Evaluate router performance on test set using fixed model instances from training.
    Splits test set into non-overlapping batches and averages results.
    
    Parameters
    ----------
    routing_df_test : pd.DataFrame
        Test routing dataframe for actual performance evaluation
        Must have columns: ['question', 'llm_name', prediction_label, 'actual_performance']
    df_model_names : list
        List of model names in correct order
    I_opt : np.ndarray
        Fixed number of instances per model (from training)
    c : np.ndarray
        Cost per query per model
    l : np.ndarray
        Concurrency per instance per model
    batch_size : int
        Number of queries per batch
    max_cost_per_query : float
        Maximum cost budget per query
    optimize_func : callable
        Optimization function (e.g., optimize_model_assignments_test)
    prediction_label : str
        Column name for predictions in the dataframe
    n_test_batches : int, optional
        Number of test batches to evaluate. If None, uses all available data
    random_state : int, optional
        Random seed for shuffling (default: 42)
    verbose : bool, optional
        Whether to print detailed progress (default: True)
    max_optimization_time_per_batch : float, optional
        If set, GLPK solver time limit (tm_lim) per mip_gap attempt in seconds.
        Each attempt gets this many seconds; up to 4 mip_gap attempts may be tried.
        If the solver does not finish in time, the incumbent is used (in-process, no subprocess).
    timeout_bounds_csv_path : str or Path, optional
        If set, batches that timed out (and used incumbent) are written to this CSV
        with columns: batch_index, relative_bound_gap, objective_lower, objective_upper,
        best_bound_reported.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'avg_pred_perf': Average predicted performance
        - 'avg_real_perf': Average real performance  
        - 'avg_perf_gap': Average performance gap
        - 'avg_cost': Average cost per query
        - 'std_pred_perf': Std dev of predicted performance
        - 'std_real_perf': Std dev of real performance
        - 'std_perf_gap': Std dev of performance gap
        - 'n_batches': Number of batches evaluated
        - 'n_queries': Total queries evaluated
        - 'model_usage': Counter of model assignments
        - 'batch_results': Raw per-batch results
    """
    if verbose:
        print("\n" + "="*70)
        print("TESTING ON TEST SET WITH FIXED INSTANCES FROM TRAIN")
        print("="*70)
    
    # Validate inputs
    if routing_df_test is None:
        raise ValueError("routing_df_test must be provided")
    
    # Create test matrix from routing_df_test
    test_matrix = routing_df_test.pivot(
        index='question',
        columns='llm_name',
        values=prediction_label
    )
    # Reorder columns to match df_model_names
    test_matrix = test_matrix[df_model_names]
    
    if verbose:
        print(f"\nRouting dataframe shape: {routing_df_test.shape}")
        print(f"Test matrix shape: {test_matrix.shape}")
    
    if verbose:
        print(f"Test matrix shape: {test_matrix.shape}")
    
    # Budget per batch
    n_test_samples = batch_size
    C_test = max_cost_per_query * n_test_samples
    
    # Shuffle test matrix once to create non-overlapping batches
    test_matrix_shuffled = test_matrix.sample(frac=1, random_state=random_state).reset_index(drop=False)
    
    if n_test_batches is None:
        # Calculate maximum number of complete batches to use entire test set
        n_test_batches = len(test_matrix_shuffled) // n_test_samples
    
    if verbose:
        print(f"\nShuffled test matrix: {test_matrix_shuffled.shape}")
        print(f"Maximum number of complete batches: {n_test_batches}")
        if max_optimization_time_per_batch is not None:
            print(f"Per-batch timeout: {max_optimization_time_per_batch:.0f}s (batches exceeding this will be skipped)")
        print(f"\nRunning optimization on {n_test_batches} non-overlapping test batches...")
        print(f"Batch size: {n_test_samples}")
        print(f"Total test queries used: {n_test_batches * n_test_samples}")
        print(f"Using I_opt from train data: {I_opt}")
    
    # Store results for each batch
    batch_results = {
        'pred_perf': [],
        'real_perf': [],
        'proportion_zero_actual': [],
        'perf_gap': [],
        'cost': [],
        'model_assignments': []
    }
    
    # Initialize timing tracking and timeout-bound rows for CSV
    total_optimization_time = 0.0
    optimization_count = 0
    timeout_bounds_rows = []

    for batch_idx in range(n_test_batches):
        # Take consecutive non-overlapping batch
        start_idx = batch_idx * n_test_samples
        end_idx = start_idx + n_test_samples
        test_sample = test_matrix_shuffled.iloc[start_idx:end_idx]
        
        # Convert to numpy array (drop question column)
        a_test_set = test_sample.drop(columns=['question']).values
        
        # Run optimization on test batch with timing (with optional timeout)
        start_time = time.time()
        timed_out = False
        if max_optimization_time_per_batch is not None:
            # In-process: GLPK tm_lim per mip_gap attempt; incumbent used when limit hit.
            x_opt_test, timed_out, obj_value, best_bound, mip_gap_used = optimize_model_assignments_test_with_timeout(
                a_test_set, c, l, I_opt, C_test, max_optimization_time_per_batch, solver="scip"
            )
        else:
            x_opt_test = optimize_func(
                a=a_test_set,
                c=c,
                l=l,
                I=I_opt,
                C=C_test,
                solver="scip"
            )
            timed_out, obj_value, best_bound, mip_gap_used = False, None, None, None
        end_time = time.time()
        batch_optimization_time = end_time - start_time

        if x_opt_test is None:
            if verbose:
                if timed_out:
                    print(f"  Batch {batch_idx+1:3d}: skipped (no solution within {max_optimization_time_per_batch:.0f}s)")
                else:
                    print(f"  Batch {batch_idx+1:3d}: skipped (optimization failed)")
            continue

        if timed_out and verbose:
            # Used incumbent; print relative difference between upper and lower bound if available
            mip_str = f", mip_gap={mip_gap_used*100:.1f}%" if mip_gap_used is not None else ""
            if obj_value is not None and best_bound is not None:
                denom = max(abs(best_bound), 1e-10)
                rel_diff = abs(best_bound - obj_value) / denom
                print(f"  Batch {batch_idx+1:3d}: timed out (used incumbent{mip_str}); relative bound gap: {rel_diff:.4f} (lower={obj_value:.6f}, upper={best_bound:.6f})")
                timeout_bounds_rows.append({
                    'batch_index': batch_idx + 1,
                    'relative_bound_gap': rel_diff,
                    'objective_lower': obj_value,
                    'objective_upper': best_bound,
                    'best_bound_reported': True,
                })
            else:
                obj_str = f", objective={obj_value:.6f}" if obj_value is not None else ""
                print(f"  Batch {batch_idx+1:3d}: timed out (used incumbent{obj_str}{mip_str}; solver did not report best bound)")
                timeout_bounds_rows.append({
                    'batch_index': batch_idx + 1,
                    'relative_bound_gap': np.nan,
                    'objective_lower': obj_value,
                    'objective_upper': np.nan,
                    'best_bound_reported': False,
                })

        total_optimization_time += batch_optimization_time
        optimization_count += 1

        # Predicted performance
        pred_perf = np.sum(a_test_set * x_opt_test) / len(a_test_set)
        cost = np.sum(x_opt_test * c) / len(a_test_set)
        
        # Evaluate real performance on test set
        test_sample_df = test_sample.reset_index()
        
        results_test = evaluate_real_router_performance(
            x_opt=x_opt_test,
            routing_df=routing_df_test,
            df_model_names=df_model_names,
            df_test=test_sample_df,
            prediction_label=prediction_label
        )
        
        # Store results
        batch_results['pred_perf'].append(pred_perf)
        batch_results['real_perf'].append(results_test['avg_actual_performance'])
        batch_results['proportion_zero_actual'].append(results_test['proportion_zero_actual'])
        batch_results['perf_gap'].append(results_test['performance_gap'])
        batch_results['cost'].append(cost)
        batch_results['model_assignments'].extend(results_test['assigned_models'])
        
        if verbose:
            print(f"  Batch {batch_idx+1:3d}: Pred={pred_perf:.4f}, Real={results_test['avg_actual_performance']:.4f}, Gap={results_test['performance_gap']:+.4f}")

    n_batches_evaluated = len(batch_results['pred_perf'])
    n_queries_evaluated = optimization_count * n_test_samples

    # Calculate aggregate statistics
    avg_pred_perf = np.mean(batch_results['pred_perf']) if batch_results['pred_perf'] else np.nan
    avg_real_perf = np.mean(batch_results['real_perf']) if batch_results['real_perf'] else np.nan
    avg_proportion_zero_actual = np.mean(batch_results['proportion_zero_actual']) if batch_results['proportion_zero_actual'] else np.nan
    avg_perf_gap = np.mean(batch_results['perf_gap']) if batch_results['perf_gap'] else np.nan
    avg_cost = np.mean(batch_results['cost']) if batch_results['cost'] else np.nan

    std_pred_perf = np.std(batch_results['pred_perf']) if len(batch_results['pred_perf']) > 1 else (0.0 if batch_results['pred_perf'] else np.nan)
    std_real_perf = np.std(batch_results['real_perf']) if len(batch_results['real_perf']) > 1 else (0.0 if batch_results['real_perf'] else np.nan)
    std_perf_gap = np.std(batch_results['perf_gap']) if len(batch_results['perf_gap']) > 1 else (0.0 if batch_results['perf_gap'] else np.nan)
    
    # Model usage statistics
    from collections import Counter
    model_counts = Counter(batch_results['model_assignments'])
    
    if verbose:
        print("\n" + "="*70)
        print("AVERAGED RESULTS OVER ALL TEST BATCHES")
        print("="*70)
        print(f"Number of test batches: {n_test_batches}")
        print(f"Batches evaluated (included): {n_batches_evaluated}")
        print(f"Total queries evaluated: {n_queries_evaluated}")
        if n_batches_evaluated < n_test_batches:
            print(f"  (skipped {n_test_batches - n_batches_evaluated} batches)")
        print(f"\nOptimization timing:")
        print(f"  Total optimization time: {total_optimization_time:.2f} seconds")
        print(f"  Number of optimizations: {optimization_count}")
        if optimization_count > 0:
            print(f"  Average time per batch:  {total_optimization_time/optimization_count:.4f} seconds")
        print(f"\nAverage predicted performance: {avg_pred_perf:.4f} ± {std_pred_perf:.4f}")
        print(f"Average real performance:      {avg_real_perf:.4f} ± {std_real_perf:.4f}")
        print(f"Average performance gap:       {avg_perf_gap:+.4f} ± {std_perf_gap:.4f}")
        print(f"Average cost per query:        {avg_cost:.2e}")
        if np.isfinite(avg_real_perf) and avg_real_perf != 0:
            print(f"Relative error: {(avg_perf_gap/avg_real_perf*100):.2f}%")
        print(f"\nTop 10 models used across all test batches:")
        for model, count in model_counts.most_common(10):
            pct = (count / n_queries_evaluated) * 100 if n_queries_evaluated else 0
            print(f"  {model:30s}: {count:4d} queries ({pct:5.1f}%)")
        
        print("="*70)

    if timeout_bounds_csv_path and timeout_bounds_rows:
        timeout_df = pd.DataFrame(timeout_bounds_rows)
        timeout_df.to_csv(timeout_bounds_csv_path, index=False)
        if verbose:
            print(f"\nTimeout bound gaps written to: {timeout_bounds_csv_path} ({len(timeout_bounds_rows)} batches)")
    
    # Return results dictionary
    return {
        'avg_pred_perf': avg_pred_perf,
        'avg_real_perf': avg_real_perf,
        'avg_proportion_zero_actual': avg_proportion_zero_actual,
        'avg_perf_gap': avg_perf_gap,
        'avg_cost': avg_cost,
        'std_pred_perf': std_pred_perf,
        'std_real_perf': std_real_perf,
        'std_perf_gap': std_perf_gap,
        'n_batches': n_batches_evaluated,
        'n_queries': n_queries_evaluated,
        'model_usage': model_counts,
        'batch_results': batch_results,
        'total_optimization_time': total_optimization_time,
        'optimization_count': optimization_count,
        'avg_optimization_time': total_optimization_time / optimization_count if optimization_count > 0 else 0
    }



def evaluate_real_router_performance(x_opt, routing_df, df_model_names, 
                                     prediction_label,
                                     df_test=None, test_query_indices=None, 
                                     include_model_names=True):
    """
    Evaluate real router performance using actual observed performance (optimized version).
    
    Parameters
    ----------
    x_opt : np.ndarray, shape (N, M)
        Query assignment matrix where x_opt[i, j] = 1 if query i is assigned to model j
    routing_df : pd.DataFrame
        DataFrame with columns ['question', 'llm_name', 'performance_prediction', 'actual_performance']
    df_model_names : list
        List of model names in the order they appear as columns (matching x_opt columns)
    df_test : pd.DataFrame, optional
        Test dataframe with 'question' column. If provided, uses the questions from this df.
        This is the preferred method when using randomly sampled data.
    test_query_indices : list or np.ndarray, optional
        If provided, only evaluate these query indices. Otherwise uses all queries.
        Note: Only used if df_test is not provided.
    include_model_names : bool, optional
        Whether to include the 'assigned_models' list in results (default: True).
        Set to False for faster execution when model names aren't needed.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'avg_actual_performance': Average actual performance across all assigned queries
        - 'avg_predicted_performance': Average predicted performance (for comparison)
        - 'actual_performances': Array of actual performances for each query
        - 'predicted_performances': Array of predicted performances for each query
        - 'assigned_models': List of model names assigned to each query (if include_model_names=True)
        - 'n_queries': Number of queries evaluated
    """
    
    # Get question names for filtering
    if df_test is not None:
        if 'question' in df_test.columns:
            question_names = df_test['question'].values
        else:
            question_names = df_test.index.values
        # Filter routing_df once before pivoting (much faster)
        routing_df_filtered = routing_df[routing_df['question'].isin(question_names)]
    elif test_query_indices is not None:
        # Get all unique questions first, then filter
        all_questions = routing_df['question'].unique()
        question_names = all_questions[test_query_indices]
        routing_df_filtered = routing_df[routing_df['question'].isin(question_names)]
    else:
        routing_df_filtered = routing_df
        question_names = None
    
    # Pivot both columns at once using pivot_table (more efficient)
    perf_matrix = routing_df_filtered.pivot_table(
        index='question',
        columns='llm_name',
        values=['actual_performance', prediction_label],
        aggfunc='first'  # Use first value if duplicates exist
    )
    
    # Filter df_model_names to only include models available in test data
    available_models = perf_matrix['actual_performance'].columns.tolist()
    filtered_model_names = [m for m in df_model_names if m in available_models]
    filtered_model_indices = [i for i, m in enumerate(df_model_names) if m in available_models]
    
    # Filter x_opt to only include available models
    x_opt_filtered = x_opt[:, filtered_model_indices]
    
    # Extract actual and predicted matrices (only for available models)
    actual_matrix = perf_matrix['actual_performance'][filtered_model_names]
    predicted_matrix = perf_matrix[prediction_label][filtered_model_names]
    
    # Ensure correct row order if question_names provided
    if question_names is not None:
        actual_matrix = actual_matrix.loc[question_names]
        predicted_matrix = predicted_matrix.loc[question_names]
    
    # Convert to numpy arrays (ensure float dtype to avoid "can't multiply sequence by non-int" errors)
    # Replace boolean strings before conversion (actual_performance is often 0/1 for correct/incorrect)
    bool_map = {'True': 1, 'true': 1, 'False': 0, 'false': 0, True: 1, False: 0}
    actual_matrix = actual_matrix.replace(bool_map)
    predicted_matrix = predicted_matrix.replace(bool_map)
    # Use pd.to_numeric with errors='coerce' for any remaining non-numeric values
    actual_array = actual_matrix.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
    predicted_array = predicted_matrix.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float64)
    x_opt_filtered = np.asarray(x_opt[:, filtered_model_indices], dtype=np.float64)
    
    # Ensure shapes match
    assert x_opt_filtered.shape == actual_array.shape, \
        f"Shape mismatch: x_opt_filtered {x_opt_filtered.shape} vs actual_array {actual_array.shape}"
    
    # Calculate performances using vectorized operations
    # x_opt_filtered[i, j] = 1 means query i is assigned to model j
    actual_performances = np.sum(actual_array * x_opt_filtered, axis=1)
    predicted_performances = np.sum(predicted_array * x_opt_filtered, axis=1)
    
    # Calculate statistics
    avg_actual = np.mean(actual_performances)
    avg_predicted = np.mean(predicted_performances)
    
    results = {
        'avg_actual_performance': avg_actual,
        'avg_predicted_performance': avg_predicted,
        'actual_performances': actual_performances,
        'predicted_performances': predicted_performances,
        'n_queries': len(actual_performances),
        'performance_gap': avg_predicted - avg_actual,  # How much we overestimated
        'proportion_zero_actual': np.mean(np.array(actual_performances) == 0)
    }
    
    # Optionally include assigned model names (slower operation)
    if include_model_names:
        assigned_model_indices = np.argmax(x_opt_filtered, axis=1)
        results['assigned_models'] = [filtered_model_names[idx] for idx in assigned_model_indices]
    
    return results


def print_router_evaluation(results):
    """
    Pretty print the router evaluation results.
    
    Parameters
    ----------
    results : dict
        Output from evaluate_real_router_performance()
    """
    print("="*70)
    print("ROUTER PERFORMANCE EVALUATION")
    print("="*70)
    print(f"Number of queries evaluated: {results['n_queries']}")
    print(f"\nPredicted average performance: {results['avg_predicted_performance']:.4f}")
    print(f"Actual average performance:    {results['avg_actual_performance']:.4f}")
    print(f"Performance gap (pred - actual): {results['performance_gap']:.4f}")
    print(f"Relative error: {(results['performance_gap']/results['avg_actual_performance']*100):.2f}%")
    
    # Model usage statistics
    from collections import Counter
    model_counts = Counter(results['assigned_models'])
    print(f"\nModel usage:")
    for model, count in model_counts.most_common():
        pct = (count / results['n_queries']) * 100
        print(f"  {model:30s}: {count:4d} queries ({pct:5.1f}%)")
    
    # Performance distribution
    actual_perf = results['actual_performances']
    print(f"\nActual performance distribution:")
    print(f"  Min:  {np.min(actual_perf):.4f}")
    print(f"  25%:  {np.percentile(actual_perf, 25):.4f}")
    print(f"  50%:  {np.median(actual_perf):.4f}")
    print(f"  75%:  {np.percentile(actual_perf, 75):.4f}")
    print(f"  Max:  {np.max(actual_perf):.4f}")
    print(f"  Std:  {np.std(actual_perf):.4f}")
    print("="*70)



def evaluate_single_llm_baseline(
    llm_name,
    gpu_counts,
    df_model_names,
    routing_df_test,
    c,
    g,
    l,
    closed_source_models,
    batch_size
):
    """
    Evaluate a single LLM baseline (route all queries to one model).
    
    This creates a results dictionary compatible with run_optimization_experiments()
    for easy comparison with optimized routing strategies.
    
    Parameters
    ----------
    llm_name : str
        Name of the LLM to use for all queries
    gpu_counts : list
        List of GPU budget values (for compatibility with plotting)
    df_model_names : list
        List of all model names
    routing_df_test : pd.DataFrame
        Test routing dataframe with actual performance data
        Must have columns: ['question', 'llm_name', 'actual_performance']
    c : np.ndarray
        Cost per query per model (M,)
    g : np.ndarray
        GPUs per instance for each model (M,)
    l : np.ndarray
        Query concurrency per instance for each model (M,)
    closed_source_models : list
        List of closed-source model names
    batch_size : int
        Maximum number of concurrent queries (batch size).
        This is used to calculate minimum GPU requirements.
    
    Returns
    -------
    dict
        Results dictionary with same structure as run_optimization_experiments(),
        containing single-LLM baseline metrics for each GPU budget.
    
    Notes
    -----
    For a single LLM baseline:
    - All queries are routed to the specified LLM
    - Performance and cost are constant across GPU budgets (once minimum GPUs are met)
    - This provides a baseline for comparison with optimization strategies
    """
    # Validate inputs
    if llm_name not in df_model_names:
        raise ValueError(f"LLM '{llm_name}' not found in df_model_names. Available: {df_model_names}")
    
    # Check routing_df_test has required columns
    required_cols = ['question', 'llm_name', 'actual_performance']
    missing_cols = [col for col in required_cols if col not in routing_df_test.columns]
    if missing_cols:
        raise ValueError(f"routing_df_test is missing required columns: {missing_cols}. "
                        f"Available: {routing_df_test.columns.tolist()}")
    
    model_idx = df_model_names.index(llm_name)
    
    # Get model properties
    model_cost = c[model_idx]
    model_gpus = g[model_idx]
    model_concurrency = l[model_idx]
    is_small = model_gpus <= 2
    is_opensource = llm_name not in closed_source_models
    
    print(f"\n{'='*70}")
    print(f"Evaluating Single LLM Baseline: {llm_name}")
    print(f"{'='*70}")
    print(f"Model properties:")
    print(f"  Cost per query: ${model_cost:.2e}")
    print(f"  GPUs per instance: {model_gpus}")
    print(f"  Concurrency per instance: {model_concurrency}")
    print(f"  Category: {'Small' if is_small else 'Large'}, {'Open-source' if is_opensource else 'Closed-source'}")
    
    # Evaluate on test data
    llm_test_data = routing_df_test[routing_df_test['llm_name'] == llm_name].copy()
    
    print(f"\nData availability for {llm_name}:")
    print(f"  Test records: {len(llm_test_data)}")
    
    if len(llm_test_data) == 0:
        print(f"WARNING: No test data found for {llm_name}!")
        test_performance = 0.0
        n_test_queries = 0
    else:
        # Coerce to numeric (RouterBench/CSV may have actual_performance as str or mixed)
        perf_series = pd.to_numeric(llm_test_data['actual_performance'], errors='coerce')
        test_performance = float(perf_series.mean())
        if np.isnan(test_performance):
            test_performance = 0.0
        # Get unique test questions
        test_questions = routing_df_test['question'].unique()
        n_test_queries = len(test_questions)
        print(f"  Mean test performance: {test_performance:.4f}")
        print(f"  Unique test questions: {n_test_queries}")
    
    avg_cost = model_cost
    
    print(f"\nPerformance metrics:")
    print(f"  Test performance: {test_performance:.4f}")
    print(f"  Average cost: ${avg_cost:.2e}")
    
    # Calculate minimum instances needed based on batch size (max concurrent queries)
    min_instances_needed = int(np.ceil(batch_size / model_concurrency))
    min_gpus_needed = min_instances_needed * model_gpus
    
    print(f"\nResource requirements (based on batch_size={batch_size}):")
    print(f"  Minimum instances needed: {min_instances_needed}")
    print(f"  Minimum GPUs needed: {min_gpus_needed}")
    
    # Create results dictionary with same structure as run_optimization_experiments()
    # Use a dummy cost value as key (since single LLM doesn't vary by cost constraint)
    dummy_cost_key = 1e-5
    
    results = {dummy_cost_key: {
        'performances': [],
        'real_performances': [],
        'test_performances': [],
        'costs': [],
        'gpus_used': [],
        'small_instances': [],
        'large_instances': [],
        'opensource_instances': [],
        'closedsource_instances': []
    }}
    
    print(f"\n{'='*70}")
    print(f"Evaluating across GPU budgets:")
    print(f"{'='*70}")
    
    for G_val in gpu_counts:
        if G_val >= min_gpus_needed:
            # Can handle all queries with this GPU budget
            instances_used = min_instances_needed
            gpus_used = instances_used * model_gpus
            
            # All metrics are constant (all queries go to this LLM)
            # Use test_performance for all performance metrics
            results[dummy_cost_key]['performances'].append(test_performance)
            results[dummy_cost_key]['real_performances'].append(test_performance)
            results[dummy_cost_key]['test_performances'].append(test_performance)
            results[dummy_cost_key]['costs'].append(avg_cost)
            results[dummy_cost_key]['gpus_used'].append(gpus_used)
            results[dummy_cost_key]['small_instances'].append(instances_used if is_small else 0)
            results[dummy_cost_key]['large_instances'].append(instances_used if not is_small else 0)
            results[dummy_cost_key]['opensource_instances'].append(instances_used if is_opensource else 0)
            results[dummy_cost_key]['closedsource_instances'].append(instances_used if not is_opensource else 0)
            
            print(f"  G={G_val:3d} -> Test: {test_performance:.4f}, "
                  f"Cost: {avg_cost:.2e}, GPUs: {gpus_used:.0f}, Instances: {instances_used:.0f}")
        else:
            # Not enough GPUs to handle all queries
            print(f"  G={G_val:3d} -> Insufficient GPUs (need {min_gpus_needed})")
            results[dummy_cost_key]['performances'].append(None)
            results[dummy_cost_key]['real_performances'].append(None)
            results[dummy_cost_key]['test_performances'].append(None)
            results[dummy_cost_key]['costs'].append(None)
            results[dummy_cost_key]['gpus_used'].append(None)
            results[dummy_cost_key]['small_instances'].append(None)
            results[dummy_cost_key]['large_instances'].append(None)
            results[dummy_cost_key]['opensource_instances'].append(None)
            results[dummy_cost_key]['closedsource_instances'].append(None)
    
    print(f"{'='*70}\n")
    
    return results



def run_optimization_experiments(
    gpu_counts, 
    max_cost_per_query_list, 
    routing_df_train,
    routing_df_test,
    c, 
    g, 
    l, 
    df_model_names,
    closed_source_models,
    batch_size,
    prediction_label,
):
    """
    Run optimization experiments for different GPU budgets and cost constraints.
    
    Parameters
    ----------
    gpu_counts : list
        List of GPU budget values to test
    max_cost_per_query_list : list
        List of maximum cost per query values to test
    routing_df_train : pd.DataFrame
        Training routing dataframe with actual performance data
        Must have columns: ['question', 'llm_name', prediction_label, 'actual_performance']
    routing_df_test : pd.DataFrame
        Test routing dataframe with actual performance data
        Must have columns: ['question', 'llm_name', prediction_label, 'actual_performance']
    batches : list
        List of batch indices
    c : np.ndarray
        Cost per query per model (M,)
    g : np.ndarray
        GPUs per instance for each model (M,)
    l : np.ndarray
        Query concurrency per instance for each model (M,)
    df_model_names : list
        List of model names in the correct order
    closed_source_models : list
        List of closed-source model names
    batch_size : int
        Batch size for test evaluation
    prediction_label : str
        Column name for predictions (default: 'main_model_prediction')
    
    Returns
    -------
    dict
        Results dictionary with keys for each cost value, containing:
        - performances: Predicted performance on train
        - real_performances: Actual observed performance on train
        - test_performances: Actual performance on test set
        - costs: Average cost per query
        - gpus_used: Number of GPUs used
        - small_instances: Number of small model instances
        - large_instances: Number of large model instances
        - opensource_instances: Number of open-source model instances
        - closedsource_instances: Number of closed-source model instances
    """
    # Create training matrix from routing_df_train
    routing_matrix_train = routing_df_train.pivot(
        index='question',
        columns='llm_name',
        values=prediction_label
    )
    # Reorder columns to match df_model_names
    routing_matrix_train = routing_matrix_train[df_model_names]

    # Convert to numpy array for optimization
    a_train = routing_matrix_train.values
    n_train_queries = a_train.shape[0]
    batches = create_random_batches(n_train_queries, batch_size, shuffle=False)

    
    # Create test matrix from routing_df_test
    test_matrix = routing_df_test.pivot(
        index='question',
        columns='llm_name',
        values=prediction_label
    )
    # Reorder columns to match df_model_names
    test_matrix = test_matrix[df_model_names]
    
    print(f"Training data: {n_train_queries} queries, {a_train.shape[1]} models")
    print(f"Test data: {test_matrix.shape[0]} queries, {test_matrix.shape[1]} models")
    # Store results for each C value (including real performance on train and test)
    results = {c_val: {
        'performances': [],  # Predicted performance on train
        'real_performances': [],  # Actual observed performance on train
        'test_performances': [],  # Actual performance on test set
        'proportion_zero_actual': [],
        'costs': [], 
        'gpus_used': [], 
        'small_instances': [], 
        'large_instances': [],
        'opensource_instances': [],  # Open-source model instances
        'closedsource_instances': []  # Closed-source model instances
    } for c_val in max_cost_per_query_list}
    
    # Define model groups based on GPU requirements (g vector)
    # Small models: require 1-2 GPUs, Large models: require 4-8 GPUs
    small_model_indices = [i for i, gpu_req in enumerate(g) if gpu_req <= 2]
    large_model_indices = [i for i, gpu_req in enumerate(g) if gpu_req >= 4]
    
    print(f"Small models (≤2 GPUs): {[df_model_names[i] for i in small_model_indices]}")
    print(f"Large models (≥4 GPUs): {[df_model_names[i] for i in large_model_indices]}")
    
    print("\nRunning optimization for different GPU counts and cost budgets...")
    print("=" * 70)


    
    for c_val in max_cost_per_query_list:
        print(f"\n{'='*70}")
        print(f"Cost budget per query = {c_val}")
        print(f"{'='*70}")
        
        for G_val in gpu_counts:
            print(f"  G={G_val:3d}")
            try:
                # Run optimization with this GPU count and cost budget
                x_opt_g, I_opt_g = optimize_model_assignments_train(
                    a_train, batches, c, g, l, C=c_val * n_train_queries, G=G_val
                )
                
                # Calculate predicted metrics
                avg_perf_predicted = np.sum(a_train * x_opt_g) / len(a_train)
                avg_cost = np.sum(x_opt_g * c) / len(a_train)
                gpus_actual = np.dot(g, I_opt_g)
                
                # Evaluate real performance on train set using actual observed data
                real_perf_results = evaluate_real_router_performance(
                    x_opt=x_opt_g,
                    routing_df=routing_df_train,
                    df_model_names=df_model_names,
                    df_test=routing_matrix_train,
                    prediction_label=prediction_label
                )
                
                avg_perf_real = real_perf_results['avg_actual_performance']
                
                # Evaluate on test set using the same I_opt_g
                test_results = evaluate_on_test_set(
                    routing_df_test=routing_df_test,
                    df_model_names=df_model_names,
                    I_opt=I_opt_g,
                    c=c,
                    l=l,
                    batch_size=batch_size,
                    max_cost_per_query=c_val,
                    optimize_func=optimize_model_assignments_test,
                    prediction_label=prediction_label,
                    n_test_batches=10,
                    random_state=42,
                    verbose=True
                )
                avg_perf_test = test_results['avg_real_perf']
                
                # Count instances for small vs large models
                small_instances = sum([I_opt_g[i] for i in small_model_indices])
                large_instances = sum([I_opt_g[i] for i in large_model_indices])
                
                # Count instances for open-source vs closed-source models
                opensource_instances = sum([I_opt_g[i] for i, model in enumerate(df_model_names) if model not in closed_source_models])
                closedsource_instances = sum([I_opt_g[i] for i, model in enumerate(df_model_names) if model in closed_source_models])
                
                results[c_val]['performances'].append(avg_perf_predicted)
                results[c_val]['real_performances'].append(avg_perf_real)
                results[c_val]['test_performances'].append(avg_perf_test)
                results[c_val]['proportion_zero_actual'].append(test_results['avg_proportion_zero_actual'])
                results[c_val]['costs'].append(avg_cost)
                results[c_val]['gpus_used'].append(gpus_actual)
                results[c_val]['small_instances'].append(small_instances)
                results[c_val]['large_instances'].append(large_instances)
                results[c_val]['opensource_instances'].append(opensource_instances)
                results[c_val]['closedsource_instances'].append(closedsource_instances)
                
                print(f"  G={G_val:3d} -> Train: {avg_perf_real:.4f}, Test: {avg_perf_test:.4f}, Cost: {avg_cost:.2e}, GPUs: {gpus_actual:.0f}, Small: {small_instances:.0f}, Large: {large_instances:.0f}")
            except Exception as e:
                print(f"  G={G_val:3d} -> Failed: {e}")
                results[c_val]['performances'].append(None)
                results[c_val]['real_performances'].append(None)
                results[c_val]['test_performances'].append(None)
                results[c_val]['proportion_zero_actual'].append(None)
                results[c_val]['costs'].append(None)
                results[c_val]['gpus_used'].append(None)
                results[c_val]['small_instances'].append(None)
                results[c_val]['large_instances'].append(None)
                results[c_val]['opensource_instances'].append(None)
                results[c_val]['closedsource_instances'].append(None)
    
    print("=" * 70)
    return results