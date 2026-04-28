import logging
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from XGBoost import XGBoost
import torch
import pickle
import argparse
from pathlib import Path
from train_mirt import load_embeddings, map_ids_to_vectors

ROOT = Path(__file__).parent.parent  # llm_routing/
MODELS_DIR = Path(__file__).parent.parent / "results" / "trained_models"
MODELS_DIR.mkdir(exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train XGBoost router model')
parser.add_argument('--use-cv', action='store_true', 
                    help='Use cross-validation to find optimal hyperparameters')
parser.add_argument('--cv-folds', type=int, default=3,
                    help='Number of cross-validation folds (default: 3)')
parser.add_argument('--search-method', type=str, default='random', choices=['grid', 'random'],
                    help='Search method for hyperparameter tuning (default: random)')
parser.add_argument('--n-iter', type=int, default=30,
                    help='Number of iterations for random search (default: 30)')
parser.add_argument('--embedding', type=str, default='bert', choices=['bert', 'open', 'zhipu', 'bge'],
                    help='Embedding type to use (default: bert)')
parser.add_argument('--bootstrap', type=int, default=1,
                    help='Number of bootstrap samples to train (default: 1, no bootstrapping)')
args = parser.parse_args()

emb_name = args.embedding
use_cross_validation = args.use_cv
n_bootstrap = args.bootstrap

if n_bootstrap > 1:
    print("="*70)
    print(f"🔄 BOOTSTRAP RESAMPLING ENABLED: {n_bootstrap} models")
    print("="*70)
    print(f"Will train {n_bootstrap} models on different bootstrap samples")
    print("="*70 + "\n")


train_data = pd.read_csv(ROOT / "data/irt_data/train.csv")

llm_embeddings, query_embeddings, llm_id_map, query_id_map = load_embeddings(emb_name)

train_llm, train_query = map_ids_to_vectors(train_data, llm_embeddings, query_embeddings, llm_id_map, query_id_map)

batch_size = 64  # Reduced to prevent memory issues during CV

llm_dim = next(iter(llm_embeddings.values())).shape[0]
query_dim = next(iter(query_embeddings.values())).shape[0]

# XGBoost-specific parameters
xgb_params = {
    'objective': 'reg:logistic',  # Logistic regression for values in [0,1]
    'eval_metric': 'logloss',  # Log loss for probabilistic predictions
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 800, # 300
    'subsample': 1.0, # 0.8,
    'colsample_bytree': 0.8, # 1.0
    'tree_method': 'hist',
    'random_state': 42,
    "gamma": 0.2,
    'min_child_weight': 5, # 1
    'n_jobs': 1,  # Single-threaded to prevent system overload during CV
    'nthread': 1  # Single-threaded to prevent system overload during CV
}

# Define parameter grid for hyperparameter tuning
param_grid = {
    "learning_rate": [0.03, 0.1],
    "max_depth": [4, 6, 8],
    "n_estimators": [300, 800],
    "min_child_weight": [1, 5],
    "gamma": [0, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


logging.getLogger().setLevel(logging.INFO)

# Bootstrap training loop
for bootstrap_idx in range(n_bootstrap):        
    
    # Resample training data with replacement for bootstrap
    if n_bootstrap > 1:

        print("\n" + "="*70)
        print(f"🔄 BOOTSTRAP ITERATION {bootstrap_idx + 1}/{n_bootstrap}")
        print("="*70)

        # Bootstrap resample: sample unique questions with replacement
        unique_questions = train_data['question'].unique()
        bootstrap_questions = np.random.choice(unique_questions, size=len(unique_questions), replace=True)
        train_data_bootstrap = train_data[train_data['question'].isin(bootstrap_questions)].reset_index(drop=True)
        print(f"Resampled {len(unique_questions)} unique questions with replacement")
        print(f"Training on {len(train_data_bootstrap)} total examples")
        
        # Get embeddings for bootstrap sample
        train_llm_bootstrap, train_query_bootstrap = map_ids_to_vectors(
            train_data_bootstrap, llm_embeddings, query_embeddings, llm_id_map, query_id_map
        )
        
        # Create DataLoader for bootstrap sample
        train_set = DataLoader(TensorDataset(
            torch.tensor(train_llm_bootstrap, dtype=torch.float32),
            torch.tensor(train_query_bootstrap, dtype=torch.float32),
            torch.tensor(train_data_bootstrap["performance"].values, dtype=torch.float32)
        ), batch_size=batch_size, shuffle=True)
    else:
        # No bootstrap - use original training data
        train_set = DataLoader(TensorDataset(
            torch.tensor(train_llm, dtype=torch.float32),
            torch.tensor(train_query, dtype=torch.float32),
            torch.tensor(train_data["performance"].values, dtype=torch.float32)
        ), batch_size=batch_size, shuffle=True)
    
    # Initialize model
    cdm = XGBoost(llm_input_dim=llm_dim, item_input_dim=query_dim, xgb_params=xgb_params)
    
    if use_cross_validation:
        print("\n" + "="*70)
        print("STEP 1: Finding Optimal Hyperparameters via Cross-Validation")
        print("="*70)
        
        # Find optimal hyperparameters using cross-validation
        cv_results = cdm.find_optimal_hyperparameters(
            train_data=train_set,
            param_grid=param_grid,
            cv=args.cv_folds,
            scoring='neg_log_loss',  # Log loss for continuous targets in (0,1)
            search_method=args.search_method,
            n_iter=args.n_iter,
            verbose=True
        )
        
        print("\n" + "="*70)
        print("STEP 2: Training Final Model with Optimized Hyperparameters")
        print("="*70)
        
        # Train with optimized hyperparameters (no need to specify epoch, it's in xgb_params now)
        cdm.train(train_set, device="cpu")
    else:
        print("\n" + "="*70)
        print("Training Model with Predefined Hyperparameters")
        print("="*70)
        
        # Train with predefined hyperparameters
        cdm.train(train_set, epoch=200)
    
    print("\n" + "="*70)
    print("Saving Model")
    print("="*70)
    
    # Save the trained model with bootstrap suffix if applicable
    if n_bootstrap > 1:
        model_name = f"xgboost_{emb_name}_bootstrap_queries_{bootstrap_idx}.snapshot"
    else:
        model_name = f"xgboost_{emb_name}.snapshot"
    
    cdm.save(MODELS_DIR / model_name)
    print(f"Model saved to {MODELS_DIR / model_name}")

if n_bootstrap > 1:
    print("\n" + "="*70)
    print(f"✅ BOOTSTRAP TRAINING COMPLETE - {n_bootstrap} models trained")
    print("="*70)

