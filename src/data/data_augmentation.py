import numpy as np
import cv2
from albumentations import (
    Compose, RandomBrightnessContrast, RandomGamma, GaussNoise, 
    ShiftScaleRotate, Blur
)
import logging

logger = logging.getLogger(__name__)

def apply_augmentation(image):
    if image.ndim != 3 or image.shape[2] != 3:
        logger.error(f"Invalid image shape: {image.shape}. Expected (H, W, 3)")
        return image

    try:
        augmentation = Compose([
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            Blur(blur_limit=3, p=0.5),
        ])
        
        augmented = augmentation(image=image)
        return augmented['image']
    except Exception as e:
        logger.error(f"Error in apply_augmentation: {str(e)}")
        return image

def augment_dataset(X, y, num_augmented=1):
    logger.info(f"Augmenting dataset. Original shape: {X.shape}")
    X_augmented = []
    y_augmented = []
    
    for image, label in zip(X, y):
        X_augmented.append(image)
        y_augmented.append(label)
        
        for _ in range(num_augmented):
            aug_image = apply_augmentation(image)
            X_augmented.append(aug_image)
            y_augmented.append(label)
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    logger.info(f"Augmented dataset shape: {X_augmented.shape}")
    return X_augmented, y_augmented