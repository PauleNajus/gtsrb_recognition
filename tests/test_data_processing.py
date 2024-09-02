import pytest
import numpy as np
from src.data.data_processing import preprocess_image, extract_features
from src.data.data_augmentation import apply_augmentation

def test_preprocess_image():
    image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    processed_image = preprocess_image(image)
    
    assert processed_image.shape == (32, 32, 3)
    assert processed_image.dtype == np.float32
    assert np.max(processed_image) <= 1.0
    assert np.min(processed_image) >= 0.0

def test_extract_features():
    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    
    features = extract_features(image)
    
    assert features.shape == (1568,)

def test_apply_augmentation():
    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    
    augmented_image = apply_augmentation(image)
    
    assert augmented_image.shape == image.shape
    assert not np.array_equal(augmented_image, image)