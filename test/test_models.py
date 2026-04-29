import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from train import MIRT, XGBoost
import torch
import pickle
import argparse

# Base path for data files (to support running from different working directories)
BASE_PATH = Path(__file__).parent
MODELS_PATH = BASE_PATH.parent / "results" / "trained_models"
DATA_PATH = BASE_PATH.parent / "data" / "irt_data"

config = {
    'glm_4_air': {
        'input_cost': 0.137e-6,
        'output_cost': 0.137e-6
    },
    'glm_4_flash': {
        'input_cost': 0.0137e-6,
        'output_cost': 0.0137e-6
    },
    'glm_4_plus': {
        'input_cost': 6.85e-6,
        'output_cost': 6.85e-6
    },
    'gpt_4o': {
        'input_cost': 2.5e-6,
        'output_cost': 10e-6
    },
    'gpt_4o_mini': {
        'input_cost': 0.15e-6,
        'output_cost': 0.6e-6
    },
    'gpt_4o_mini_cot': {
        'input_cost': 0.15e-6,
        'output_cost': 0.6e-6
    },
    'deepseek_coder': {
        'input_cost': 0.14e-6,
        'output_cost': 0.28e-6
    },
    'deepseek_chat': {
        'input_cost': 0.14e-6,
        'output_cost': 0.28e-6
    },
    'qwen25_32b_int4': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'qwen25_7b_instruct': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'qwen25_72b_instruct': {
        'input_cost': 1.08e-6,
        'output_cost': 1.08e-6
    },
    'qwq_32b_preview': {
        'input_cost': 1.2e-6,
        'output_cost': 1.2e-6
    },
    'qwen25_math_7b_instruct': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'llama31_8b_instruct': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'llama31_70b_instruct': {
        'input_cost': 0.792e-6,
        'output_cost': 0.792e-6
    },
    'llama31_405b_instruct': {
        'input_cost': 3.15e-6,
        'output_cost': 3.15e-6
    },
    'mixtral_8x7b_instruct': {
        'input_cost': 0.54e-6,
        'output_cost': 0.54e-6
    },
    'mistral_7b_instruct_v02': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'ministral_8b_instruct_2410': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'gemini15_flash': {
        'input_cost': 0.075e-6,
        'output_cost': 0.3e-6
    },
    'claude35_haiku20241022': {
        'input_cost': 0.8e-6,
        'output_cost': 4e-6
    },
}


llms = [
    'glm_4_air',
    'glm_4_flash',
    'glm_4_plus',
    'gpt_4o',
    'gpt_4o_mini',
    'gpt_4o_mini_cot',
    'deepseek_coder',
    'deepseek_chat',
    'qwen25_32b_int4',
    'qwen25_7b_instruct',
    'qwen25_72b_instruct',
    'qwq_32b_preview',
    'qwen25_math_7b_instruct',
    'llama31_8b_instruct',
    'llama31_70b_instruct',
    'llama31_405b_instruct',
    'mixtral_8x7b_instruct',
    'mistral_7b_instruct_v02',
    'ministral_8b_instruct_2410',
    'gemini15_flash',
]

def load_embeddings(embeddings_dir):
    with open(BASE_PATH.parent / f"utils/{embeddings_dir}_embeddings/llm_embeddings.pkl", "rb") as f:
        llm_embeddings_data = pickle.load(f)
    llm_embeddings = {llm["index"]: np.array(llm["embedding"], dtype=np.float32) for llm in llm_embeddings_data}

    with open(BASE_PATH.parent / f"utils/{embeddings_dir}_embeddings/query_embeddings.pkl", "rb") as f:
        query_embeddings_data = pickle.load(f)
    query_embeddings = {query["index"]: np.array(query["embedding"], dtype=np.float32) for query in query_embeddings_data}

    with open(BASE_PATH.parent / f"utils/relevance/relevance_vectors_cluster_test_{embeddings_dir}.pkl", "rb") as f:
        relevance_embeddings_data = pickle.load(f)
    relevance_embeddings = {relevance["index"]: np.array(relevance["relevance_vector"], dtype=np.float32) for relevance in relevance_embeddings_data}

    with open(BASE_PATH.parent / f"utils/cold/test_avg_embeddings_{embeddings_dir}.pkl", "rb") as f:
        cold_embeddings_data = pickle.load(f)
    cold_embeddings = {cold["index"]: np.array(cold["avg_embedding"], dtype=np.float32) for cold in cold_embeddings_data}
    
    llm_id_map = pd.read_csv(BASE_PATH.parent / f"utils/map/llm.csv", index_col="name").to_dict()["index"]
    query_id_map = pd.read_csv(BASE_PATH.parent / f"utils/map/query.csv", index_col="question").to_dict()["index"]
    return llm_embeddings, query_embeddings, cold_embeddings, relevance_embeddings, llm_id_map, query_id_map



def compute_xgboost_bootstrap_performance_estimates(emb_name, test_path, task=None, lamda=0.0,
                                     n_bootstrap=100, candidate_llms=None):
    """
    Pre-compute XGBoost bootstrap routing for all test questions and candidate LLMs.
    
    Args:
        emb_name: Embedding name (e.g., 'bert', 'open')
        test_path: Path to test data file
        task: Optional task filter
        lamda: Lambda parameter for mixing query and cold vectors
        n_bootstrap: Number of bootstrap models to load
        candidate_llms: List of candidate LLMs to consider (None = all)
    
    Returns:
        DataFrame with columns: question, llm_name, llm_cost, performance_prediction,
        actual_performance, main_model_prediction, bootstrap_mean_prediction,
        bootstrap_predictions, bootstrap_quantile_2_5, bootstrap_quantile_5,
        bootstrap_quantile_95, bootstrap_quantile_97_5
    """
    print(f"Computing XGBoost bootstrap routing with {n_bootstrap} bootstrap models...")
    
    # Load embeddings
    llm_embeddings, query_embeddings, cold_embeddings, relevance_embeddings, llm_id_map, query_id_map = load_embeddings(emb_name)
    
    # Determine dimensions
    dim_map = {"open": 1536, "zhipu": 512, "bge": 1024, "bert": 768}
    llm_dim = query_dim = dim_map.get(emb_name, 768)

    # Load test data
    test_data = pd.read_csv(DATA_PATH / f"{test_path}.csv")
    if task is not None:
        test_data = test_data[test_data['task'] == task]
        print(f"Filtered to task: {task}")
    
    if candidate_llms is None:
        candidate_llms = llms
    else:
        invalid_llms = [llm for llm in candidate_llms if llm not in llms]
        if invalid_llms:
            raise ValueError(f"Invalid LLM names: {invalid_llms}. Valid options: {llms}")
    
    print(f"Using candidate LLMs: {candidate_llms}")
    
    # Load main model
    print(f"Loading main XGBoost model and {n_bootstrap} bootstrap models...")
    cdm_main = XGBoost(llm_dim, query_dim)
    cdm_main.load(str(MODELS_PATH / f"xgboost_{emb_name}.snapshot"))
    
    # Load all bootstrap models
    cdm_models = []
    for i in range(n_bootstrap):
        model = XGBoost(llm_dim, query_dim)
        model.load(str(MODELS_PATH / f"xgboost_{emb_name}_bootstrap_queries_{i}.snapshot"))
        cdm_models.append(model)
    print(f"Successfully loaded main model and {len(cdm_models)} bootstrap models")
    
    # Create performance lookup
    performance_lookup = {(row['question'], row['llm']): row['performance'] 
                         for _, row in test_data.iterrows()}
    
    # Get unique questions
    unique_questions = test_data['question'].unique()
    print(f"Processing {len(unique_questions)} unique questions across {len(candidate_llms)} LLMs...")
    
    # Storage for routing records
    routing_records = []
    
    # Process each question
    for q_idx, question in enumerate(unique_questions):
        # Print progress every 50 questions
        if (q_idx + 1) % 50 == 0:
            print(f"Processed {q_idx + 1}/{len(unique_questions)} questions...")
        
        # Process each LLM
        for llm_name in candidate_llms:
            # Get vectors
            llm_vector, query_vector, cold_vector, relevance_vector = map_ids_to_vectors(
                llm_name, question, llm_embeddings, query_embeddings, 
                cold_embeddings, relevance_embeddings, llm_id_map, query_id_map
            )
            
            # Mix query and cold vectors
            query_vector = (1 - lamda) * query_vector + lamda * cold_vector
            
            # Get main model prediction
            main_pred = cdm_main.generate(torch.Tensor(llm_vector), torch.Tensor(query_vector))
            
            # Get all bootstrap predictions
            bootstrap_preds = []
            for model in cdm_models:
                pred = model.generate(torch.Tensor(llm_vector), torch.Tensor(query_vector))
                bootstrap_preds.append(pred)
            
            bootstrap_preds = np.array(bootstrap_preds)
            
            # Compute statistics
            bootstrap_mean = np.mean(bootstrap_preds)
            quantile_2_5 = np.percentile(bootstrap_preds, 2.5)
            quantile_5 = np.percentile(bootstrap_preds, 5)
            quantile_95 = np.percentile(bootstrap_preds, 95)
            quantile_97_5 = np.percentile(bootstrap_preds, 97.5)
            
            # Get actual performance
            actual_performance = performance_lookup.get((question, llm_name), None)
            
            # Create record
            record = {
                'question': question,
                'llm_name': llm_name,
                'llm_cost': config[llm_name]["output_cost"],
                'performance_prediction': quantile_2_5,  # Using quantile_2_5 as default prediction
                'actual_performance': actual_performance,
                'main_model_prediction': main_pred,
                'bootstrap_mean_prediction': bootstrap_mean,
                'bootstrap_predictions': bootstrap_preds.tolist(),
                'bootstrap_quantile_2_5': quantile_2_5,
                'bootstrap_quantile_5': quantile_5,
                'bootstrap_quantile_95': quantile_95,
                'bootstrap_quantile_97_5': quantile_97_5
            }
            
            routing_records.append(record)
    
    print(f"Completed! Generated {len(routing_records)} routing records.")

    routing_df = pd.DataFrame(routing_records)

    save_dir = BASE_PATH.parent / "results" / "performance_estimates"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"xgboost_bootstrap_{n_bootstrap}_{emb_name}_{test_path}.csv"
    routing_df.to_csv(save_path, index=False)
    print(f"Saved routing dataframe to {save_path}")

    return routing_df


def compute_knn_bootstrap_performance_estimates(emb_name, test_path, task=None, k_neighbors=5, 
                                   n_bootstrap=100, candidate_llms=None):
    """
    Compute k-NN bootstrap routing predictions for all questions and LLMs.
    
    This is a self-contained method that:
    - Loads training and test data
    - Loads embeddings
    - Pre-computes bootstrap samples
    - Computes k-NN predictions on full data and bootstrap samples
    
    Args:
        emb_name: Embedding type ('bert', 'open', etc.)
        test_path: Path to test data (e.g., 'test1', 'test2', 'train')
        task: Optional task filter (e.g., 'hotpot_qa')
        k_neighbors: Number of nearest neighbors (default: 5)
        n_bootstrap: Number of bootstrap samples (default: 100)
        candidate_llms: List of LLM names to consider (default: all available)
    
    Returns:
        DataFrame with routing records for each (question, llm) pair
    """
    print(f"Loading data for k-NN bootstrap routing...")
    
    # Load test data
    test_data = pd.read_csv(DATA_PATH / f"{test_path}.csv")
    
    # Filter by task if specified
    if task is not None:
        test_data = test_data[test_data['task'] == task]
        print(f"Filtered to task: {task}")
    
    unique_questions = test_data['question'].unique()
    print(f"Total unique questions: {len(unique_questions)}")
    
    # Load embeddings
    llm_embeddings, query_embeddings, cold_embeddings, relevance_embeddings, llm_id_map, query_id_map = load_embeddings(emb_name)
    
    # Use candidate_llms if provided, otherwise use all available llms
    if candidate_llms is None:
        candidate_llms = llms
    else:
        # Validate that all provided llms are valid
        invalid_llms = [llm for llm in candidate_llms if llm not in llms]
        if invalid_llms:
            raise ValueError(f"Invalid LLM names provided: {invalid_llms}. Valid options are: {llms}")
    
    print(f"Using candidate LLMs: {candidate_llms}")
    
    # Load training data
    print(f"Loading training data with k={k_neighbors}...")
    train_data = pd.read_csv(DATA_PATH / "train.csv")
    
    # Create lookup for training performances: (question, llm) -> performance
    train_performance_lookup = {(row['question'], row['llm']): row['performance'] 
                               for _, row in train_data.iterrows()}
    
    # Get unique training questions
    train_questions = train_data['question'].unique()
    print(f"Loaded {len(train_questions)} unique training questions")
    
    # Create performance lookup for test data
    performance_lookup = {(row['question'], row['llm']): row['performance']
                         for _, row in test_data.iterrows()}
    
    # Pre-compute valid questions and vectors for regular k-NN
    print(f"Pre-computing valid training vectors...")
    knn_valid_questions = []
    knn_valid_vectors = []
    for train_question in train_questions:
        train_query_vector = query_embeddings.get(query_id_map.get(train_question))
        if train_query_vector is not None:
            knn_valid_questions.append(train_question)
            knn_valid_vectors.append(train_query_vector)
    knn_valid_vectors = np.array(knn_valid_vectors)
    print(f"Pre-computed {len(knn_valid_questions)} valid training vectors")
    
    # Pre-compute bootstrap samples and their valid vectors
    print(f"k-NN Bootstrap: Pre-computing {n_bootstrap} bootstrap samples...")
    bootstrap_train_samples = []
    for _ in range(n_bootstrap):
        # Bootstrap sample of training questions (sample with replacement)
        bootstrap_sample = np.random.choice(
            train_questions, 
            size=len(train_questions), 
            replace=True
        )
        
        # Pre-extract valid questions and vectors for this bootstrap sample
        valid_questions = []
        valid_vectors = []
        for train_question in bootstrap_sample:
            train_query_vector = query_embeddings.get(query_id_map.get(train_question))
            if train_query_vector is not None:
                valid_questions.append(train_question)
                valid_vectors.append(train_query_vector)
        
        # Store as numpy array for fast vectorized operations
        if len(valid_vectors) > 0:
            bootstrap_train_samples.append({
                'questions': valid_questions,
                'vectors': np.array(valid_vectors)
            })
    
    print(f"k-NN Bootstrap: Finished pre-computing {len(bootstrap_train_samples)} bootstrap samples")
    
    # Now compute routing predictions
    print("Computing k-NN bootstrap predictions for all question-LLM pairs...")
    routing_records = []
    
    for q_idx, question in enumerate(unique_questions):
        # Print progress every 50 questions
        if (q_idx + 1) % 50 == 0:
            print(f"Processed {q_idx + 1}/{len(unique_questions)} questions...")
        
        # Get query embedding for this question
        query_vector = query_embeddings.get(query_id_map.get(question))
        if query_vector is None:
            continue
        
        # OPTIMIZATION: Compute distances ONCE per question (not per LLM!)
        # Distances only depend on query, not on LLM
        
        # Compute distances to full training data
        distances_full = np.linalg.norm(knn_valid_vectors - query_vector, axis=1)
        k_indices_full = np.argpartition(distances_full, min(k_neighbors, len(distances_full)-1))[:k_neighbors]
        k_neighbor_questions_full = [knn_valid_questions[idx] for idx in k_indices_full]
        
        # Compute distances for each bootstrap sample
        bootstrap_k_neighbors = []
        for bootstrap_sample in bootstrap_train_samples:
            valid_questions = bootstrap_sample['questions']
            valid_vectors = bootstrap_sample['vectors']
            
            # Vectorized distance computation
            distances = np.linalg.norm(valid_vectors - query_vector, axis=1)
            k_indices = np.argpartition(distances, min(k_neighbors, len(distances)-1))[:k_neighbors]
            k_neighbor_questions = [valid_questions[idx] for idx in k_indices]
            bootstrap_k_neighbors.append(k_neighbor_questions)
        
        # Now for each LLM, look up performances for the k-nearest neighbors
        for llm_name in candidate_llms:
            # Get performances for k nearest neighbors on full data
            neighbor_performances_full = []
            for neighbor_question in k_neighbor_questions_full:
                perf = train_performance_lookup.get((neighbor_question, llm_name))
                if perf is not None:
                    neighbor_performances_full.append(perf)
            
            # Main prediction from full training data
            if len(neighbor_performances_full) > 0:
                main_pred = np.mean(neighbor_performances_full)
            else:
                main_pred = 0.0
            
            # Get performances for k nearest neighbors in each bootstrap sample
            bootstrap_knn_preds = []
            for k_neighbor_questions in bootstrap_k_neighbors:
                neighbor_performances = []
                for neighbor_question in k_neighbor_questions:
                    perf = train_performance_lookup.get((neighbor_question, llm_name))
                    if perf is not None:
                        neighbor_performances.append(perf)
                
                # Average the performances for this bootstrap sample
                if len(neighbor_performances) > 0:
                    bootstrap_pred = np.mean(neighbor_performances)
                    bootstrap_knn_preds.append(bootstrap_pred)
            
            # Convert to numpy array
            bootstrap_knn_preds = np.array(bootstrap_knn_preds)
            
            # Compute statistics from bootstrap distribution
            if len(bootstrap_knn_preds) > 0:
                bootstrap_mean = np.mean(bootstrap_knn_preds)

                # Calculate quantiles
                quantile_2_5 = np.percentile(bootstrap_knn_preds, 2.5)
                quantile_5 = np.percentile(bootstrap_knn_preds, 5)
                quantile_95 = np.percentile(bootstrap_knn_preds, 95)
                quantile_97_5 = np.percentile(bootstrap_knn_preds, 97.5)
                
                # Use quantile as performance prediction
                performance_pred = quantile_2_5
            else:
                bootstrap_mean = 0.0
                std_pred = 0.0
                quantile_2_5 = 0.0
                quantile_5 = 0.0
                quantile_95 = 0.0
                quantile_97_5 = 0.0
                performance_pred = main_pred if main_pred > 0 else 0.0
            
            # Get actual observed performance
            actual_performance = performance_lookup.get((question, llm_name), None)
            
            # Create routing record
            record = {
                'question': question,
                'llm_name': llm_name,
                'llm_cost': config[llm_name]["output_cost"],
                'performance_prediction': performance_pred,
                'actual_performance': actual_performance,
                'main_model_prediction': main_pred,
                'bootstrap_mean_prediction': bootstrap_mean,
                'bootstrap_predictions': bootstrap_knn_preds.tolist() if len(bootstrap_knn_preds) > 0 else [],
                'bootstrap_quantile_2_5': quantile_2_5,
                'bootstrap_quantile_5': quantile_5,
                'bootstrap_quantile_95': quantile_95,
                'bootstrap_quantile_97_5': quantile_97_5
            }
            
            routing_records.append(record)
    
    print(f"Computed routing for {len(routing_records)} question-LLM pairs")

    # Convert to DataFrame
    routing_df = pd.DataFrame(routing_records)

    save_dir = BASE_PATH.parent / "results" / "performance_estimates"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"knn_{k_neighbors}_bootstrap_{n_bootstrap}_{emb_name}_{test_path}.csv"
    routing_df.to_csv(save_path, index=False)
    print(f"Saved routing dataframe to {save_path}")

    return routing_df


def map_ids_to_vectors(llm_name, question, llm_embeddings, query_embeddings, cold_embeddings, relevance_embeddings, llm_id_map, query_id_map):
    llm_id = llm_id_map[llm_name]
    query_id = query_id_map[question]
    return np.array(llm_embeddings[llm_id]), np.array(query_embeddings[query_id]), np.array(cold_embeddings.get(query_id, np.zeros_like(query_embeddings[query_id]))), np.array(relevance_embeddings.get(query_id, np.ones(25)))


def load_train_query_embeddings(emb_name):
    """Load training query embeddings and training data (same approach as compute_pairwise_distances.py)"""
    train_data = pd.read_csv(DATA_PATH / "train.csv")
    
    with open(BASE_PATH.parent / f"utils/{emb_name}_embeddings/query_embeddings.pkl", "rb") as f:
        query_embeddings_data = pickle.load(f)
    query_embeddings = {query["index"]: np.array(query["embedding"], dtype=np.float32) for query in query_embeddings_data}
    
    query_id_map = pd.read_csv(BASE_PATH.parent / f"utils/map/query.csv", index_col="question").to_dict()["index"]
    
    # Get unique training questions
    train_questions = train_data['question'].unique()
    
    # Collect embeddings for training questions
    train_embeddings = []
    for question in train_questions:
        if question in query_id_map:
            query_id = query_id_map[question]
            if query_id in query_embeddings:
                train_embeddings.append(query_embeddings[query_id])
    
    train_embedding_matrix = np.array(train_embeddings)
    
    # Check for zero vectors (same as in compute_pairwise_distances.py)
    norms = np.linalg.norm(train_embedding_matrix, axis=1)
    zero_vectors = norms < 1e-10
    if np.any(zero_vectors):
        print(f"WARNING: Found {np.sum(zero_vectors)} zero or near-zero vectors in training data")
        train_embedding_matrix = train_embedding_matrix[~zero_vectors]
        print(f"After removing zero vectors: {train_embedding_matrix.shape}")
    
    return train_embedding_matrix



def compute_mirt_performance_estimates(emb_name, test_path, task=None, lamda=0.0, candidate_llms=None):
    """
    Compute MIRT performance predictions for all test questions and candidate LLMs.

    Args:
        emb_name: Embedding name (e.g., 'bert', 'open')
        test_path: Test split name (e.g., 'test1')
        task: Optional task filter
        lamda: Cold-start mixing weight
        candidate_llms: LLMs to evaluate (None = all)

    Returns:
        DataFrame with columns: question, llm_name, llm_cost,
        performance_prediction, actual_performance
    """
    if candidate_llms is None:
        candidate_llms = llms
    else:
        invalid = [m for m in candidate_llms if m not in llms]
        if invalid:
            raise ValueError(f"Invalid LLM names: {invalid}. Valid options: {llms}")

    test_data = pd.read_csv(DATA_PATH / f"{test_path}.csv")
    if task is not None:
        test_data = test_data[test_data['task'] == task]
        print(f"Filtered to task: {task}")

    performance_lookup = {(r['question'], r['llm']): r['performance'] for _, r in test_data.iterrows()}

    llm_embeddings, query_embeddings, cold_embeddings, relevance_embeddings, llm_id_map, query_id_map = load_embeddings(emb_name)

    dim_map = {"open": 1536, "zhipu": 512, "bge": 1024, "bert": 768}
    llm_dim = query_dim = dim_map.get(emb_name, 768)

    cdm = MIRT(llm_dim, query_dim, 25)
    cdm.load(str(MODELS_PATH / f"mirt_{emb_name}.snapshot"))
    print("Loaded MIRT model")

    unique_questions = test_data['question'].unique()
    print(f"Processing {len(unique_questions)} unique questions across {len(candidate_llms)} LLMs...")

    routing_records = []
    for q_idx, question in enumerate(unique_questions):
        if (q_idx + 1) % 50 == 0:
            print(f"Processed {q_idx + 1}/{len(unique_questions)} questions...")
        for llm_name in candidate_llms:
            llm_vec, q_vec, cold_vec, _ = map_ids_to_vectors(
                llm_name, question, llm_embeddings, query_embeddings,
                cold_embeddings, relevance_embeddings, llm_id_map, query_id_map,
            )
            q_vec = (1 - lamda) * q_vec + lamda * cold_vec
            perf_pred = cdm.generate(torch.Tensor(llm_vec), torch.Tensor(q_vec))
            routing_records.append({
                'question': question,
                'llm_name': llm_name,
                'llm_cost': config[llm_name]["output_cost"],
                'performance_prediction': perf_pred,
                'actual_performance': performance_lookup.get((question, llm_name)),
            })

    print(f"Completed! Generated {len(routing_records)} routing records.")
    routing_df = pd.DataFrame(routing_records)

    save_dir = BASE_PATH.parent / "results" / "performance_estimates"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"mirt_{emb_name}_{test_path}.csv"
    routing_df.to_csv(save_path, index=False)
    print(f"Saved routing dataframe to {save_path}")

    return routing_df


def evaluate_routing(routing_df, test_path, a, task=None, candidate_llms=None,
                     prediction_label='performance_prediction'):
    """
    Select the best LLM per question from a precomputed performance-estimates DataFrame
    using the objective: a * prediction - (1-a) * cost.

    Args:
        routing_df: DataFrame from any compute_*_performance_estimates function.
            Required columns: question, llm_name, llm_cost, and prediction_label.
        test_path: Test split name (e.g., 'test1') — used to load ground-truth costs.
        a: Performance weight in [0, 1].
        task: Optional task filter applied to both routing_df and ground-truth data.
        candidate_llms: LLMs to consider (None = all in routing_df).
        prediction_label: Column to use as the performance signal.

    Returns:
        avg_performance, total_cost, opt_cost, chosen_llms
    """
    test_data = pd.read_csv(DATA_PATH / f"{test_path}.csv")
    if task is not None:
        test_data = test_data[test_data['task'] == task]
        valid_qs = test_data['question'].unique()
        routing_df = routing_df[routing_df['question'].isin(valid_qs)]

    if candidate_llms is None:
        candidate_llms = routing_df['llm_name'].unique().tolist()

    performance_lookup = {(r['question'], r['llm']): r['performance'] for _, r in test_data.iterrows()}
    input_tokens_lookup = {(r['question'], r['llm']): r['input_tokens'] for _, r in test_data.iterrows()}
    output_tokens_lookup = {(r['question'], r['llm']): r['output_tokens'] for _, r in test_data.iterrows()}

    required_cols = ['question', 'llm_name', 'llm_cost']
    missing = [c for c in required_cols if c not in routing_df.columns]
    if missing:
        raise ValueError(f"routing_df missing columns: {missing}")
    if prediction_label not in routing_df.columns and 'performance_prediction' not in routing_df.columns:
        raise ValueError(f"Column '{prediction_label}' not found in routing_df")

    b = -(1 - a)
    final_performance, final_cost, chosen_llms_out, opt_cost = [], [], [], []

    for q_idx, question in enumerate(routing_df['question'].unique()):
        if (q_idx + 1) % 1000 == 0:
            print(f"Processing question {q_idx + 1}...")

        records = routing_df[routing_df['question'] == question].to_dict('records')
        if not records:
            continue

        best_record, best_score = None, float('-inf')
        for rec in records:
            perf_pred = rec.get(prediction_label) if prediction_label in rec else rec.get('performance_prediction')
            if perf_pred is None:
                raise ValueError(f"Column '{prediction_label}' not found in routing_df")
            score = a * perf_pred + b * config[rec['llm_name']]["output_cost"] * 1e5
            if score > best_score:
                best_score, best_record = score, rec

        if best_record is None:
            continue
        best_llm = best_record['llm_name']
        perf = performance_lookup.get((question, best_llm))
        in_t = input_tokens_lookup.get((question, best_llm))
        out_t = output_tokens_lookup.get((question, best_llm))
        if perf is None or in_t is None or out_t is None:
            continue

        final_performance.append(perf)
        final_cost.append(in_t * config[best_llm]["input_cost"] + out_t * config[best_llm]["output_cost"])
        chosen_llms_out.append(best_llm)

        best_opt_perf, best_opt_llm = -1, None
        for llm in candidate_llms:
            q_perf = performance_lookup.get((question, llm))
            if q_perf is not None and q_perf > best_opt_perf:
                best_opt_perf, best_opt_llm = q_perf, llm
        if best_opt_llm is not None:
            oi = input_tokens_lookup.get((question, best_opt_llm))
            oo = output_tokens_lookup.get((question, best_opt_llm))
            opt_cost.append(
                oi * config[best_opt_llm]["input_cost"] + oo * config[best_opt_llm]["output_cost"]
                if oi and oo else
                in_t * config[best_llm]["input_cost"] + out_t * config[best_llm]["output_cost"]
            )
        else:
            opt_cost.append(in_t * config[best_llm]["input_cost"] + out_t * config[best_llm]["output_cost"])

    avg_performance = np.mean(final_performance)
    total_cost = np.sum(final_cost)
    opt_cost_total = np.sum(opt_cost)
    print(f"Performance: {avg_performance}  Total Cost: {total_cost}  Opt Cost: {opt_cost_total}")
    return avg_performance, total_cost, opt_cost_total, chosen_llms_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=str, required=True, choices=["mirt", "xgboost", "knn"],
                        help="Router type: mirt | xgboost | knn")
    parser.add_argument("--emb_name", type=str, default="bert")
    parser.add_argument("--test_path", type=str, default="test1")
    parser.add_argument("--lamda", type=float, default=0.0, help="Cold-start mixing weight (mirt/xgboost)")
    parser.add_argument("--n_bootstrap", type=int, default=100, help="Bootstrap samples (xgboost/knn)")
    parser.add_argument("--k_neighbors", type=int, default=40, help="k nearest neighbours (knn only)")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--candidate_llms", type=str, nargs='+', default=None)
    args = parser.parse_args()

    if args.router == "mirt":
        compute_mirt_performance_estimates(
            emb_name=args.emb_name,
            test_path=args.test_path,
            task=args.task,
            lamda=args.lamda,
            candidate_llms=args.candidate_llms,
        )
    elif args.router == "xgboost":
        compute_xgboost_bootstrap_performance_estimates(
            emb_name=args.emb_name,
            test_path=args.test_path,
            task=args.task,
            lamda=args.lamda,
            n_bootstrap=args.n_bootstrap,
            candidate_llms=args.candidate_llms,
        )
    elif args.router == "knn":
        compute_knn_bootstrap_performance_estimates(
            emb_name=args.emb_name,
            test_path=args.test_path,
            task=args.task,
            k_neighbors=args.k_neighbors,
            n_bootstrap=args.n_bootstrap,
            candidate_llms=args.candidate_llms,
        )