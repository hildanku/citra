import numpy as np
import cv2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import argparse

def load_image(filepath):
    return cv2.imread(filepath)

def rgb_to_hsi(img):
    img = img.astype(np.float32) / 255.0
    R, G, B = cv2.split(img)

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(num / (den + 1e-6))
    
    H = np.copy(theta)
    H[B > G] = 2 * np.pi - H[B > G]
    H = H / (2 * np.pi)

    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - 3 * min_RGB / (R + G + B + 1e-6)
    I = (R + G + B) / 3.0

    return H, S, I

def compute_features(img):
    H, S, I = rgb_to_hsi(img)
    features = {
        'MeanR': np.mean(img[:,:,0]),
        'MeanG': np.mean(img[:,:,1]),
        'MeanB': np.mean(img[:,:,2]),
        'MeanH': np.mean(H),
        'MeanS': np.mean(S),
        'MeanI': np.mean(I),
        'VarRed': np.var(img[:,:,0]),
        'VarGreen': np.var(img[:,:,1]),
        'VarBlue': np.var(img[:,:,2]),
        'VarH': np.var(H),
        'VarS': np.var(S),
        'VarI': np.var(I),
        'RangeR': np.max(img[:,:,0]) - np.min(img[:,:,0]),
        'RangeG': np.max(img[:,:,1]) - np.min(img[:,:,1]),
        'RangeB': np.max(img[:,:,2]) - np.min(img[:,:,2]),
        'RangeH': np.max(H) - np.min(H),
        'RangeS': np.max(S) - np.min(S),
        'RangeI': np.max(I) - np.min(I),
        'ContrastR': np.std(img[:,:,0]) / np.mean(img[:,:,0]),
        'ContrastG': np.std(img[:,:,1]) / np.mean(img[:,:,1]),
        'ContrastB': np.std(img[:,:,2]) / np.mean(img[:,:,2]),
        'ContrastH': np.std(H) / np.mean(H),
        'ContrastS': np.std(S) / np.mean(S),
        'ContrastI': np.std(I) / np.mean(I)
    }
    return features

def train_knn_classifier(training_data_path):
    training_data = pd.read_excel(training_data_path)
    X = training_data.iloc[:, :-1].values
    y = training_data.iloc[:, -1].values
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

def classify_image(knn, img_features):
    img_features = np.array(list(img_features.values())).reshape(1, -1)
    result = knn.predict(img_features)[0]
    return ["MATURE", "HALF-MATURE", "IMMATURE"][result - 1]

def main():
    parser = argparse.ArgumentParser(description="Image Classification using KNN")
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('training_data_path', type=str, help='Path to the training data Excel file')
    args = parser.parse_args()

    img = load_image(args.image_path)
    img_features = compute_features(img)
    knn = train_knn_classifier(args.training_data_path)
    classification_result = classify_image(knn, img_features)

    print(f"Classification Result: {classification_result}")

if __name__ == "__main__":
    main()
