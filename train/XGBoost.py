import logging
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import pickle


def custom_log_loss(y_true, y_pred):
    """
    Compute log loss for continuous targets in (0,1).
    For XGBRegressor with objective='reg:logistic', predictions are already in [0,1].
    
    Log loss formula: -[y * log(p) + (1-y) * log(1-p)]
    
    Parameters
    ----------
    y_true : array-like
        True labels (continuous values in [0,1])
    y_pred : array-like
        Predicted probabilities (in [0,1])
    
    Returns
    -------
    float
        Log loss value
    """
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute log loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


class XGBoost(): 
    """
    XGBoost-based replacement for MIRT.
    """
    
    def __init__(self, llm_input_dim=None, item_input_dim=None, xgb_params=None):
        """
        Initialize XGBoost-based model.
        
        Parameters
        ----------
        llm_input_dim : int, optional
            Dimension of LLM embeddings (kept for interface compatibility)
        item_input_dim : int, optional
            Dimension of item embeddings (kept for interface compatibility)
        xgb_params : dict, optional
            XGBoost parameters. If None, uses default parameters.
        """
        super(XGBoost, self).__init__()
        
        self.xgb_params = xgb_params
        self.model = None
        self.llm_input_dim = llm_input_dim
        self.item_input_dim = item_input_dim

    def _prepare_data(self, data_loader):
        """
        Convert PyTorch DataLoader batches into concatenated feature matrix and labels.
        
        Parameters
        ----------
        data_loader : DataLoader
            PyTorch DataLoader with batches of (llm, item, response)
        
        Returns
        -------
        X : np.ndarray
            Feature matrix (concatenated llm and item embeddings)
        y : np.ndarray
            Labels (responses)
        """
        X_list = []
        y_list = []
        
        for batch_data in tqdm(data_loader, "Preparing data"):
            llm, item, response = batch_data
            
            # Convert to numpy if they're torch tensors
            if hasattr(llm, 'numpy'):
                llm = llm.cpu().numpy()
            if hasattr(item, 'numpy'):
                item = item.cpu().numpy()
            if hasattr(response, 'numpy'):
                response = response.cpu().numpy()
            
            # Concatenate llm and item embeddings
            batch_features = np.concatenate([llm, item], axis=1)
            
            X_list.append(batch_features)
            y_list.append(response)
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        return X, y

    def find_optimal_hyperparameters(self, train_data, param_grid=None, cv=5, 
                                    scoring='neg_log_loss', search_method='grid',
                                    n_iter=50, verbose=True):
        """
        Find optimal XGBoost hyperparameters using cross-validation.
        
        Parameters
        ----------
        train_data : DataLoader
            Training data loader
        param_grid : dict, optional
            Dictionary with parameters names as keys and lists of parameter settings to try.
            If None, uses a default parameter grid.
        cv : int
            Number of cross-validation folds
        scoring : str
            Scoring metric to optimize. Options: 'neg_log_loss', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'
        search_method : str
            'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        n_iter : int
            Number of parameter settings sampled for RandomizedSearchCV
        verbose : bool
            Whether to print progress
        
        Returns
        -------
        dict
            Best parameters found
        """
        print("Preparing training data for hyperparameter tuning...")
        X_train, y_train = self._prepare_data(train_data)
        
        # Create custom scorer for log loss if needed
        if scoring == 'neg_log_loss':
            scoring = make_scorer(custom_log_loss, greater_is_better=False)
        
        # Base model with fixed parameters
        base_model = xgb.XGBRegressor(
            objective='reg:logistic',  # Logistic regression for values in [0,1]
            eval_metric='logloss',  # Log loss for probabilistic predictions
            tree_method='hist',
            random_state=42,
            device='cpu'
        )
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                verbose=2 if verbose else 0,
                n_jobs=1  # Single-threaded to prevent system overload
            )
            print(f"Starting Grid Search with {cv}-fold cross-validation...")
        elif search_method == 'random':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                verbose=2 if verbose else 0,
                n_jobs=1,  # Single-threaded to prevent system overload
                random_state=42
            )
            print(f"Starting Randomized Search with {n_iter} iterations and {cv}-fold cross-validation...")
        else:
            raise ValueError(f"Unknown search_method: {search_method}. Use 'grid' or 'random'.")
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Update model parameters with best found
        best_params = search.best_params_
        self.xgb_params.update(best_params)
        
        print("\n" + "="*60)
        print("Hyperparameter Tuning Results")
        print("="*60)
        print(f"Best Score ({scoring}): {search.best_score_:.6f}")
        print(f"Best Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print("="*60 + "\n")
        
        # Return results
        return {
            'best_params': best_params,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }

    def train(self, train_data, *, epoch: int = None, lr=0.001, verbose=True) -> None:
        """
        Train the XGBoost model.
        
        Parameters
        ----------
        train_data : DataLoader
            Training data loader
        epoch : int, optional
            Number of boosting rounds (overrides n_estimators if provided)
        lr : float
            Learning rate (overrides the one in xgb_params)
        verbose : bool
            Whether to print progress
        """
        # Update parameters if provided
        if epoch is not None:
            self.xgb_params['n_estimators'] = epoch
        if lr != 0.001:  # If non-default learning rate provided
            self.xgb_params['learning_rate'] = lr
        
        print("Preparing training data...")
        X_train, y_train = self._prepare_data(train_data)
        
        print("Training XGBoost model...")
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_train, y_train, verbose=verbose)
        
        print("Training completed!")


    def generate(self, llm, item):
        """
        Generate predictions for given llm and item embeddings.
        
        Parameters
        ----------
        llm : torch.Tensor or np.ndarray
            LLM embeddings
        item : torch.Tensor or np.ndarray
            Item embeddings
        
        Returns
        -------
        list
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Convert to numpy if needed
        if hasattr(llm, 'numpy'):
            llm = llm.cpu().numpy()
        if hasattr(item, 'numpy'):
            item = item.cpu().numpy()
        
        # Check if single sample or batch
        single_sample = False
        if llm.ndim == 1:
            llm = llm.reshape(1, -1)
            single_sample = True
        if item.ndim == 1:
            item = item.reshape(1, -1)
        
        # Concatenate features
        X = np.concatenate([llm, item], axis=1)
        
        # Get predictions
        pred = self.model.predict(X)
        
        # Return scalar for single sample, list for batch (matching MIRT behavior)
        if single_sample and len(pred) == 1:
            return float(pred[0])
        return pred.tolist()
    

    def save(self, filepath):
        """
        Save the XGBoost model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Save both the model and the parameters
        save_dict = {
            'model': self.model,
            'xgb_params': self.xgb_params,
            'llm_input_dim': self.llm_input_dim,
            'item_input_dim': self.item_input_dim,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logging.info(f"Saved XGBoost model to {filepath}")

    def load(self, filepath):
        """
        Load the XGBoost model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model = save_dict['model']
        self.xgb_params = save_dict['xgb_params']
        self.llm_input_dim = save_dict.get('llm_input_dim')
        self.item_input_dim = save_dict.get('item_input_dim')
        
        logging.info(f"Loaded XGBoost model from {filepath}")

