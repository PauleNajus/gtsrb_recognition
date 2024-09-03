import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from src.data.data_augmentation import augment_dataset
from config import Config
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_path, target_size=(32, 32)):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] != 3:
        raise ValueError(f"Unexpected number of channels: {img.shape[2]}")

    img = cv2.resize(img, target_size)
    return img.astype(np.float32) / 255.0

def extract_features(image):
    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), channel_axis=-1)
    
    hist = cv2.calcHist([(image * 255).astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    return np.concatenate([hog_features, hist])

def load_dataset(csv_paths, image_dir, augment=True):
    data, labels = [], []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep=';')
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row['Filename'])
            if os.path.exists(img_path):
                data.append(preprocess_image(img_path))
                labels.append(int(row['ClassId']))
            else:
                logger.warning(f"Image not found: {img_path}")
    
    X, y = np.array(data), np.array(labels)
    logger.info(f"Original dataset shape: X={X.shape}, y={y.shape}")
    
    if augment:
        X, y = augment_dataset(X, y)
        logger.info(f"Dataset shape after augmentation: X={X.shape}, y={y.shape}")
    
    return X, y

def load_and_preprocess_data():
    csv_paths = Config.get_train_csv_paths()
    logger.info(f"CSV paths: {csv_paths}")
    X, y = load_dataset(csv_paths, Config.TRAIN_IMAGES_PATH)
    logger.info(f"Loaded data shape: X={X.shape}, y={y.shape}")
    return X, y

def load_test_data():
    test_df = pd.read_csv(Config.TEST_GT_PATH, sep=';')
    X_test, y_test = [], []
    
    for _, row in test_df.iterrows():
        img_path = os.path.join(Config.TEST_IMAGES_PATH, row['Filename'])
        if os.path.exists(img_path):
            X_test.append(preprocess_image(img_path))
            y_test.append(row['ClassId'])
        else:
            logger.warning(f"Image file not found: {img_path}")
    
    X_test, y_test = np.array(X_test), np.array(y_test)
    logger.info(f"Loaded {len(X_test)} test images")
    
    return X_test, y_test