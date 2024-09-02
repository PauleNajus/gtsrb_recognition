import optuna
import torch
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)

def objective_rf(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42
    )

    return cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='accuracy').mean()

def tune_random_forest(X, y, n_trials=20):
    study = optuna.create_study(direction='maximize')

    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)
    
    X_sample = np.array(X_sample)
    y_sample = np.array(y_sample)
    
    if X_sample.ndim > 2:
        X_sample = X_sample.reshape(X_sample.shape[0], -1)
    
    study.optimize(lambda trial: objective_rf(trial, X_sample, y_sample), n_trials=n_trials)
    
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    best_rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        n_jobs=-1,
        random_state=42
    )

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    
    best_rf.fit(X, y)
    
    return best_rf, best_params

class GPURandomForestPredictor:
    def __init__(self, rf_model):
        self.n_estimators = rf_model.n_estimators
        self.n_classes = rf_model.n_classes_
        self.trees = [self._tree_to_cuda(tree) for tree in rf_model.estimators_]

    def _tree_to_cuda(self, tree):
        return {
            'children_left': torch.tensor(tree.tree_.children_left, device='cuda'),
            'children_right': torch.tensor(tree.tree_.children_right, device='cuda'),
            'feature': torch.tensor(tree.tree_.feature, device='cuda'),
            'threshold': torch.tensor(tree.tree_.threshold, device='cuda'),
            'value': torch.tensor(tree.tree_.value, device='cuda'),
        }

    def predict(self, X):
        X_cuda = torch.tensor(X, dtype=torch.float32, device='cuda')
        predictions = torch.zeros((X.shape[0], self.n_classes), device='cuda')
        
        for tree in self.trees:
            node_indicator = torch.zeros(X.shape[0], dtype=torch.long, device='cuda')
            
            while True:
                feature = tree['feature'][node_indicator]
                threshold = tree['threshold'][node_indicator]
                
                go_left = X_cuda[torch.arange(X.shape[0]), feature] <= threshold
                node_indicator[go_left] = tree['children_left'][node_indicator[go_left]]
                node_indicator[~go_left] = tree['children_right'][node_indicator[~go_left]]
                
                if (node_indicator == -1).all():
                    break
            
            predictions += tree['value'][node_indicator, 0]
        
        return predictions / self.n_estimators