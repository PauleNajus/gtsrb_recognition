import pytest
import torch
from src.models.neural_network import TrafficSignCNN
from src.models.machine_learning import create_random_forest

def test_traffic_sign_cnn():
    model = TrafficSignCNN()

    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    
    assert output.shape == (1, 43)

def test_random_forest():
    rf = create_random_forest()
    
    assert rf.n_estimators == 100
    assert rf.random_state is not None