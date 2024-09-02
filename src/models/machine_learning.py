from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearnex import patch_sklearn
patch_sklearn()
import torch

def create_random_forest(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42
    )

def tune_random_forest(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

class PyTorchRandomForest(torch.nn.Module):
    def __init__(self, n_estimators, max_depth, n_features, n_classes):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_classes = n_classes

        self.trees = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(n_features, 2**max_depth),
                torch.nn.ReLU(),
                torch.nn.Linear(2**max_depth, n_classes)
            ) for _ in range(n_estimators)
        ])
    
    def forward(self, x):
        return torch.stack([tree(x) for tree in self.trees]).mean(dim=0)

def create_random_forest(n_estimators=100, max_depth=5, n_features=None, n_classes=None):
    return PyTorchRandomForest(n_estimators, max_depth, n_features, n_classes)