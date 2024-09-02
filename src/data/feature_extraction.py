import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, multichannel=True)
    return features

def extract_features(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    hog_features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), channel_axis=-1)

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    pixel_features = image.flatten()

    features = np.concatenate([hog_features, hist, pixel_features])

    if len(features) < 3072:
        features = np.pad(features, (0, 3072 - len(features)))
    elif len(features) > 3072:
        features = features[:3072]
    
    return features