import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from sklearn.ensemble import RandomForestClassifier
import random

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equalized = cv2.equalizeHist(img)
    return cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def detect_minutiae(image_path, display=False):
    processed_image = preprocess_image(image_path)
    harris_response = cv2.cornerHarris(processed_image, blockSize=2, ksize=3, k=0.04)
    harris_response = cv2.dilate(harris_response, None)
    _, harris_response = cv2.threshold(harris_response, 0.01 * harris_response.max(), 255, 0)
    harris_response = np.uint8(harris_response)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    minutiae = cv2.goodFeaturesToTrack(harris_response, maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=3, useHarrisDetector=True)
    if display:
        image_copy = processed_image.copy()
        for minutia in minutiae:
            x, y = minutia.ravel()
            cv2.circle(image_copy, (x, y), 3, 255, -1)
        plt.imshow(image_copy)
        plt.show()
    return minutiae

def extract_features(minutiae, fixed_size=500):
    features = np.array([minutia.ravel() for minutia in minutiae if minutia is not None]).flatten()
    if len(features) > fixed_size:
        features = features[:fixed_size]
    elif len(features) < fixed_size:
        features = np.pad(features, (0, fixed_size - len(features)), 'constant')
    return features

def knn_classifier(train_features, train_labels):
    classifier = KNeighborsClassifier(n_neighbors=100)
    classifier.fit(train_features, train_labels)
    return classifier

def svm_classifier(train_features, train_labels):
    classifier = SVC(gamma='scale', kernel='rbf', probability=True)
    classifier.fit(train_features, train_labels)
    return classifier

def random_forest_classifier(train_features, train_labels):
    classifier = RandomForestClassifier(n_estimators=100, max_depth=200, min_samples_split=10, random_state=42)
    classifier.fit(train_features, train_labels)
    return classifier

def evaluate_models(knn_model, svm_model, rf_model, test_features, test_labels):
    # Code to evaluate the performance of the models
    # ...
    return accuracy, report, frr, far

def compare_fingerprints(knn_model, svm_model, rf_model, image_a_path, image_b_path, similarity_threshold=0.27, debug=False):
    minutiae_a = detect_minutiae(image_a_path, debug)
    minutiae_b = detect_minutiae(image_b_path, debug)
    features_a = extract_features(minutiae_a)
    features_b = extract_features(minutiae_b)
    combined_features = np.concatenate((features_a, features_b))
    # Code to compare the fingerprints using the models
    # ...
    return True if similarity == 1 else False

def main():
    # Code to load and process the fingerprint images
    # ...
    knn_model = knn_classifier(train_features, train_labels)
    svm_model = svm_classifier(train_features, train_labels)
    rf_model = random_forest_classifier(train_features, train_labels)
    accuracy, report, frr, far = evaluate_models(knn_model, svm_model, rf_model, test_features, test_labels)
    # Print the results
    # ...

if __name__ == '__main__':
    main()
