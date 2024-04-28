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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equalized = cv2.equalizeHist(image)
    return cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def detect_minutiae(image_path, display=False):
    processed_image = preprocess_image(image_path)
    corners = cv2.cornerHarris(processed_image, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    _, corners = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)
    corners = np.uint8(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    minutiae = cv2.goodFeaturesToTrack(corners, maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=3, useHarrisDetector=True)
    if display:
        display_image = processed_image.copy()
        for point in minutiae:
            x, y = point.ravel()
            cv2.circle(display_image, (x, y), 3, 255, -1)
        plt.imshow(display_image)
        plt.show()
    return minutiae

def extract_features(minutiae, fixed_size=500):
    features = np.array([point.ravel() for point in minutiae if point is not None]).flatten()
    if len(features) > fixed_size:
        features = features[:fixed_size]
    elif len(features) < fixed_size:
        features = np.pad(features, (0, fixed_size - len(features)), 'constant')
    return features

def train_knn_classifier(train_features, train_labels):
    classifier = KNeighborsClassifier(n_neighbors=100)
    classifier.fit(train_features, train_labels)
    return classifier

def train_svm_classifier(train_features, train_labels):
    classifier = SVC(gamma='scale', kernel='rbf', probability=True)
    classifier.fit(train_features, train_labels)
    return classifier

def train_random_forest_classifier(train_features, train_labels):
    classifier = RandomForestClassifier(n_estimators=100, max_depth=200, min_samples_split=10, random_state=42)
    classifier.fit(train_features, train_labels)
    return classifier

def evaluate_performance(knn_classifier, svm_classifier, rf_classifier, test_features, test_labels):
    max_frr = 0
    min_frr = 1
    max_far = 0
    min_far = 1
    avg_frr = 0
    avg_far = 0
    eer = 1

    for threshold in range(1, 99):
        knn_predictions = knn_classifier.predict_proba(test_features)[:, 1]
        knn_predictions = [x * 4 for x in knn_predictions]
        svm_predictions = svm_classifier.predict_proba(test_features)[:, 1]
        svm_predictions = [x * 2 for x in svm_predictions]
        rf_predictions = rf_classifier.predict_proba(test_features)[:, 1]
        rf_predictions = [x * 4 for x in rf_predictions]
        combined_predictions = np.add(knn_predictions, svm_predictions)
        combined_predictions = np.add(combined_predictions, rf_predictions)
        predictions = ((combined_predictions[:] / 10) >= threshold / 100).astype(int)

        sum_frr = 0
        sum_far = 0
        true_rejects = 0
        true_accepts = 0

        for i in range(len(test_labels)):
            if test_labels[i] == 1:
                true_accepts += 1
            else:
                true_rejects += 1
            if test_labels[i] == 1 and predictions[i] == 0:
                sum_frr += 1
            if test_labels[i] == 0 and predictions[i] == 1:
                sum_far += 1

        sub_avg_frr = sum_frr / true_accepts
        sub_avg_far = sum_far / true_rejects

        if sub_avg_frr > max_frr:
            max_frr = sub_avg_frr
        if sub_avg_frr < min_frr:
            min_frr = sub_avg_frr
        if sub_avg_far > max_far:
            max_far = sub_avg_far
        if sub_avg_far < min_far:
            min_far = sub_avg_far

        if (sub_avg_frr - 0.02) <= sub_avg_far and sub_avg_far <= (sub_avg_frr + 0.02):
            if (sub_avg_frr + sub_avg_far) / 2 < eer:
                eer = (sub_avg_frr + sub_avg_far) / 2

    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, zero_division=0)

    avg_frr /= 99
    avg_far /= 99

    return accuracy, report, max_frr, min_frr, avg_frr, max_far, min_far, avg_far, eer

def main():
    image_dir = 'NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/full-data'
    reference_image_paths = glob.glob(f'{image_dir}/f*.png')
    subject_image_paths = {os.path.basename(path).split('_')[0][1:]: path for path in glob.glob(f'{image_dir}/s*.png')}
    reference_image_paths.sort()

    paired_features = []
    labels = []

    for i in range(len(reference_image_paths) - 500):
        ref_path = reference_image_paths[i]
        file_id = os.path.basename(ref_path).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            ref_minutiae = detect_minutiae(ref_path, disp=False)
            subj_minutiae = detect_minutiae(subj_path, disp=False)
            ref_features = extract_features(ref_minutiae)
            subj_features = extract_features(subj_minutiae)
            combined_features = np.concatenate((ref_features, subj_features))
            paired_features.append(combined_features)
            labels.append(1)

        if i == 0:
            dif_path = reference_image_paths[len(reference_image_paths) - 501]
            dif_path_2 = reference_image_paths[len(reference_image_paths) - 502]
            dif_path_3 = reference_image_paths[len(reference_image_paths) - 503]
        elif i == 1:
            dif_path = reference_image_paths[0]
            dif_path_2 = reference_image_paths[len(reference_image_paths) - 501]
            dif_path_3 = reference_image_paths[len(reference_image_paths) - 502]
        elif i == 2:
            dif_path = reference_image_paths[1]
            dif_path_2 = reference_image_paths[0]
            dif_path_3 = reference_image_paths[len(reference_image_paths) - 501]
        else:
            dif_path = reference_image_paths[i - 1]
            dif_path_2 = reference_image_paths[i - 2]
            dif_path_3 = reference_image_paths[i - 3]

        file_id = os.path.basename(dif_path).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            subj_minutiae = detect_minutiae(subj_path, disp=False)
            subj_features = extract_features(subj_minutiae)
            combined_features = np.concatenate((ref_features, subj_features))
            paired_features.append(combined_features)
            labels.append(0)

        file_id = os.path.basename(dif_path_2).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            subj_minutiae = detect_minutiae(subj_path, disp=False)
            subj_features = extract_features(subj_minutiae)
            combined_features = np.concatenate((ref_features, subj_features))
            paired_features.append(combined_features)
            labels.append(0)

        file_id = os.path.basename(dif_path_3).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            subj_minutiae = detect_minutiae(subj_path, disp=False)
            subj_features = extract_features(subj_minutiae)
            combined_features = np.concatenate((ref_features, subj_features))
            paired_features.append(combined_features)
            labels.append(0)

    train_features = np.array(paired_features)
    train_labels = np.array(labels, dtype=int)
    test_features = []
    test_labels = []

    for i in range(500):
        ref_path = reference_image_paths[i + 1500]
        ref_minutiae = detect_minutiae(ref_path, disp=False)
        ref_features = extract_features(ref_minutiae)
        k = random.randint(0, 1)
        if k == 0:
            file_id = os.path.basename(ref_path).split('_')[0][1:]
            subj_path = subject_image_paths.get(file_id)
            if subj_path:
                subj_minutiae = detect_minutiae(subj_path, disp=False)
                subj_features = extract_features(subj_minutiae)
                combined_features = np.concatenate((ref_features, subj_features))
                test_features.append(combined_features)
                test_labels.append(1)
        else:
            j = random.randint(1500, 1999)
            while j == i:
                j = random.randint(1500, 1999)
            dif_path = reference_image_paths[j]
            file_id = os.path.basename(dif_path).split('_')[0][1:]
            subj_path = subject_image_paths.get(file_id)
            if subj_path:
                subj_minutiae = detect_minutiae(subj_path, disp=False)
                subj_features = extract_features(subj_minutiae)
                combined_features = np.concatenate((ref_features, subj_features))
                test_features.append(combined_features)
                test_labels.append(0)

    total_samples = len(train_features) + len(test_features)
    train_samples = len(train_features)
    test_samples = len(test_features)
    print(f"Total Samples: {total_samples}")
    print(f"Training Samples: {train_samples}")
    print(f"Testing Samples: {test_samples}")

    knn_classifier = train_knn_classifier(train_features, train_labels)
    svm_classifier = train_svm_classifier(train_features, train_labels)
    rf_classifier = train_random_forest_classifier(train_features, train_labels)

    hybrid_accuracy, hybrid_report, max_frr_hybrid, min_frr_hybrid, avg_frr_hybrid, max_far_hybrid, min_far_hybrid, avg_far_hybrid, eer_hybrid = evaluate_performance(
        knn_classifier, svm_classifier, rf_classifier, test_features, test_labels
    )

    print("Hybrid Accuracy: ", hybrid_accuracy)
    print("Hybrid Report:\n", hybrid_report)
    print(f"Hybrid Max FRR: {max_frr_hybrid:.4f}, Min FRR: {min_frr_hybrid:.4f}, Avg FRR: {avg_frr_hybrid:.4f}")
    print(f"Hybrid Max FAR: {max_far_hybrid:.4f}, Min FAR: {min_far_hybrid:.4f}, Avg FAR: {avg_far_hybrid:.4f}")
    print(f"Hybrid Equal Error Rate (EER): {eer_hybrid:.4f}")

    summary_table = [
        ["hybrid", hybrid_accuracy, max_frr_hybrid, min_frr_hybrid, avg_frr_hybrid, max_far_hybrid, min_far_hybrid, avg_far_hybrid, eer_hybrid]
    ]
    headers = ["Method", "Accuracy", "Max FRR", "Min FRR", "Avg FRR", "Max FAR", "Min FAR", "Avg FAR", "EER"]
    print(tabulate(summary_table, headers, tablefmt="grid"))

if __name__ == '__main__':
    main()
