import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))   # llm_routing/
sys.path.insert(0, str(Path(__file__).parent))           # llm_routing/routing/

import numpy as np

from solver.eval import evaluate_single_llm_baseline
from solver.experiments import run_router_optimization_experiments, get_baseline_results
from test.test_models import config

LLM_ROUTING_DIR = Path(__file__).parent.parent
PERF_EST_DIR = LLM_ROUTING_DIR / "results" / "performance_estimates"
RESULTS_DIR = LLM_ROUTING_DIR / "results" / "routing_results"


def create_model_configuration(model_names, verbose=True):
    """
    Build GPU, cost, and concurrency vectors for a list of model names.

    Returns a dict with keys:
        g  – GPU requirements per model (np.ndarray)
        c  – Output cost per query per model (np.ndarray)
        l  – Concurrency limit per model (np.ndarray)
        closed_source_models, gpu_requirements, concurrency_limits, model_names
    """
    gpu_requirements = {
        # 8-GPU tier
        'llama31_405b_instruct': 8,
        'deepseek_coder': 8,
        'deepseek_chat': 8,
        'gpt_4o': 8,
        'glm_4_plus': 8,
        # 4-GPU tier
        'qwq_32b_preview': 4,
        'qwen25_72b_instruct': 4,
        'llama31_70b_instruct': 4,
        'claude35_haiku20241022': 4,
        'glm_4_air': 4,
        # 2-GPU tier
        'mixtral_8x7b_instruct': 2,
        # 1-GPU tier
        'glm_4_flash': 1,
        'gpt_4o_mini': 1,
        'gpt_4o_mini_cot': 1,
        'qwen25_7b_instruct': 1,
        'qwen25_math_7b_instruct': 1,
        'qwen25_32b_int4': 1,
        'llama31_8b_instruct': 1,
        'mistral_7b_instruct_v02': 1,
        'ministral_8b_instruct_2410': 1,
        'gemini15_flash': 1,
    }

    # Conservative lower-bound concurrent decode slots per instance.
    concurrency_limits = {
        'llama31_405b_instruct': 1,
        'deepseek_coder': 2,
        'deepseek_chat': 2,
        'gpt_4o': 16,
        'glm_4_plus': 16,
        'qwq_32b_preview': 4,
        'qwen25_72b_instruct': 4,
        'llama31_70b_instruct': 4,
        'claude35_haiku20241022': 16,
        'glm_4_air': 4,
        'mixtral_8x7b_instruct': 8,
        'glm_4_flash': 16,
        'gpt_4o_mini': 16,
        'gpt_4o_mini_cot': 16,
        'qwen25_7b_instruct': 16,
        'qwen25_math_7b_instruct': 16,
        'qwen25_32b_int4': 16,
        'llama31_8b_instruct': 16,
        'mistral_7b_instruct_v02': 16,
        'ministral_8b_instruct_2410': 16,
        'gemini15_flash': 16,
    }

    closed_source_models = {
        'gpt_4o', 'glm_4_plus', 'claude35_haiku20241022',
        'gpt_4o_mini', 'gpt_4o_mini_cot', 'gemini15_flash',
    }

    # Closed-source models are API-based: 0 GPUs, non-zero cost.
    gpu_requirements_final = {
        **{k: v for k, v in gpu_requirements.items() if k not in closed_source_models},
        **{k: 0 for k in closed_source_models},
    }

    g = np.array([gpu_requirements_final.get(model, 1) for model in model_names])
    c = np.array([config[model]['output_cost'] if model in closed_source_models else 0.0
                  for model in model_names])
    l = np.array([float(concurrency_limits.get(model, 1)) for model in model_names])

    if verbose:
        print(f"\nModel configuration: {len(model_names)} models "
              f"({sum(m not in closed_source_models for m in model_names)} open-source, "
              f"{sum(m in closed_source_models for m in model_names)} closed-source)")
        for i, m in enumerate(model_names):
            print(f"  {m:45s}  GPUs={g[i]}  cost={c[i]:.2e}  concurrency={l[i]:.0f}")

    return {
        'g': g,
        'c': c,
        'l': l,
        'closed_source_models': closed_source_models,
        'gpu_requirements': gpu_requirements_final,
        'concurrency_limits': concurrency_limits,
        'model_names': model_names,
    }


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_name = 'test1'

    routers = [
        ('mirt',    'performance_prediction', test_name,
         str(PERF_EST_DIR / f'mirt_bert_{test_name}.csv')),
        ('xgboost', 'main_model_prediction',  test_name,
         str(PERF_EST_DIR / f'xgboost_bootstrap_100_bert_{test_name}.csv')),
        ('xgboost', 'bootstrap_quantile_10',  test_name,
         str(PERF_EST_DIR / f'xgboost_bootstrap_100_bert_{test_name}.csv')),
        ('knn_40',  'main_model_prediction',  test_name,
         str(PERF_EST_DIR / f'knn_40_bootstrap_100_bert_{test_name}.csv')),
        ('knn_40',  'bootstrap_quantile_10',  test_name,
         str(PERF_EST_DIR / f'knn_40_bootstrap_100_bert_{test_name}.csv')),
    ]

    max_cost_per_query_list = [1.0e-6, 3.0e-6]

    for batch_size in [100]:
        gpu_counts = np.linspace(0, batch_size, 10, dtype=int).tolist()

        run_router_optimization_experiments(
            routers=routers,
            gpu_counts=gpu_counts,
            max_cost_per_query_list=max_cost_per_query_list,
            batch_size=batch_size,
            create_model_configuration=create_model_configuration,
            train_ratio=0.1,
            seed=42,
            verbose=True,
            output_dir=RESULTS_DIR,
        )

    routing_csv_path = str(PERF_EST_DIR / f'knn_40_bootstrap_100_bert_{test_name}.csv')

    for batch_size in [100]:
        gpu_counts = np.linspace(0, batch_size, 10, dtype=int).tolist()

        get_baseline_results(
            routing_csv_path=routing_csv_path,
            gpu_counts=gpu_counts,
            batch_size=batch_size,
            baseline_csv_path=str(
                RESULTS_DIR / f'baseline_results_{batch_size}_{test_name}.csv'
            ),
            create_model_configuration=create_model_configuration,
            evaluate_single_llm_baseline=evaluate_single_llm_baseline,
            train_ratio=0.1,
            seed=42,
            verbose=True,
        )
