import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from src.data.data_augmentation import augment_dataset
from config import Config
import os
import warnings
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
    elif img.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unexpected number of channels: {img.shape[2]}")

    img = cv2.resize(img, target_size)
    
    img = img.astype(np.float32) / 255.0
    
    return img

def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    return features

def extract_color_histogram(image, bins=(8, 8, 8)):
    image_uint8 = (image * 255).astype(np.uint8)
    hist = cv2.calcHist([image_uint8], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_features(image):
    hog_features = extract_hog_features(image)
    color_hist = extract_color_histogram(image)
    return np.concatenate([hog_features, color_hist])

def load_dataset(csv_paths, image_dir, augment=True):
    data = []
    labels = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep=';')
        for _, row in df.iterrows():
            filename = row['Filename']
            class_id = row['ClassId']
            img_path = os.path.join(image_dir, os.path.dirname(csv_path).split(os.path.sep)[-1], filename)
            
            if os.path.exists(img_path):
                img = preprocess_image(img_path)
                data.append(img)
                labels.append(int(class_id))
            else:
                logger.warning(f"Image not found: {img_path}")
    
    X = np.array(data)
    y = np.array(labels)
    
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
    test_images_dir = Config.TEST_IMAGES_PATH
    test_gt_file = Config.TEST_GT_PATH
    
    if not os.path.exists(test_gt_file):
        raise FileNotFoundError(f"Test ground truth file not found: {test_gt_file}")
    
    test_df = pd.read_csv(test_gt_file, sep=';')
    X_test = []
    y_test = []
    
    for _, row in test_df.iterrows():
        img_path = os.path.join(test_images_dir, row['Filename'])
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue
        
        img = preprocess_image(img_path)
        X_test.append(img)
        y_test.append(row['ClassId'])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Loaded {len(X_test)} test images")
    
    return X_test, y_test