import numpy as np
import cv2
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
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
    preprocessed_image = preprocess_image(image_path)
    dst = cv2.cornerHarris(preprocessed_image, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dhttps://github.com/qyz9437/CSEC_472_Lab4_Biometrics.gitst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.goodFeaturesToTrack(dst, maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=3, useHarrisDetector=True)
    if display:
        display_image = preprocessed_image.copy()
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(display_image, (x, y), 3, 255, -1)
        plt.imshow(display_image)
        plt.show()
    return corners

def extract_features(minutiae, fixed_size=500):
    features = np.array([m.ravel() for m in minutiae if m is not None]).flatten()
    if len(features) > fixed_size:
        features = features[:fixed_size]
    elif len(features) < fixed_size:
        features = np.pad(features, (0, fixed_size - len(features)), 'constant')
    return features

def ml_technique_one(train_features, train_labels):
    classifier = KNeighborsClassifier(n_neighbors=100)
    classifier.fit(train_features, train_labels)
    return classifier

def ml_technique_two(train_features, train_labels):
    classifier = SVC(gamma='scale', kernel='rbf', probability=True)
    classifier.fit(train_features, train_labels)
    return classifier

def ml_technique_three(train_features, train_labels):
    classifier = RandomForestClassifier(n_estimators=100, max_depth=200, min_samples_split=10, random_state=42)
    classifier.fit(train_features, train_labels)
    return classifier

def evaluate_performance(classifier, test_features, test_labels):
    max_frr = 0
    min_frr = 1
    max_far = 0
    min_far = 1
    avg_frr = 0
    avg_far = 0
    eer = 1
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, zero_division=0)
    for threshold in range(1, 99):
        predictions = (classifier.predict_proba(test_features)[:, 1] >= (threshold / 100)).astype(int)
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
        if (sub_avg_frr - 0.07) <= sub_avg_far and sub_avg_far <= (sub_avg_frr + 0.07):
            if (sub_avg_frr + sub_avg_far) / 2 < eer:
                eer = (sub_avg_frr + sub_avg_far) / 2
        avg_frr += sub_avg_frr
        avg_far += sub_avg_far
    avg_frr = avg_frr / 99
    avg_far = avg_far / 99
    return accuracy, report, max_frr, min_frr, avg_frr, max_far, min_far, avg_far, eer

def main():
    image_dir = 'NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/full-data'
    reference_image_paths = glob.glob(f'{image_dir}/f*.png')
    subject_image_paths = {os.path.basename(p).split('_')[0][1:]: p for p in glob.glob(f'{image_dir}/s*.png')}
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
    print("\n")
    total_samples = len(train_features) + len(test_features)
    train_samples = len(train_features)
    test_samples = len(test_features)
    print(f"Total Samples: {total_samples}")
    print(f"Training Samples: {train_samples}")
    print(f"Testing Samples: {test_samples}")
    print("\n")
    knn_classifier = ml_technique_one(train_features, train_labels)
    svm_classifier = ml_technique_two(train_features, train_labels)
    rf_classifier = ml_technique_three(train_features, train_labels)
    knn_accuracy, knn_report, max_frr_knn, min_frr_knn, avg_frr_knn, max_far_knn, min_far_knn, avg_far_knn, eer_knn = evaluate_performance(knn_classifier, test_features, test_labels)
    svm_accuracy, svm_report, max_frr_svm, min_frr_svm, avg_frr_svm, max_far_svm, min_far_svm, avg_far_svm, eer_svm = evaluate_performance(svm_classifier, test_features, test_labels)
    rf_accuracy, rf_report, max_frr_rf, min_frr_rf, avg_frr_rf, max_far_rf, min_far_rf, avg_far_rf, eer_rf = evaluate_performance(rf_classifier, test_features, test_labels)
    print("KNN Accuracy: ", knn_accuracy)
    print("KNN Report:\n", knn_report)
    print(f"KNN Max FRR: {max_frr_knn:.4f}, Min FRR: {min_frr_knn:.4f}, Avg FRR: {avg_frr_knn:.4f}")
    print(f"KNN Max FAR: {max_far_knn:.4f}, Min FAR: {min_far_knn:.4f}, Avg FAR: {avg_far_knn:.4f}")
    print(f"KNN Equal Error Rate (EER): {eer_knn:.4f}")
    print("\n")
    print("SVM Accuracy: ", svm_accuracy)
    print("SVM Report:\n", svm_report)
    print(f"SVM Max FRR: {max_frr_svm:.4f}, Min FRR: {min_frr_svm:.4f}, Avg FRR: {avg_frr_svm:.4f}")
    print(f"SVM Max FAR: {max_far_svm:.4f}, Min FAR: {min_frr_svm:.4f}, Avg FAR: {avg_far_svm:.4f}")
    print(f"SVM Equal Error Rate (EER): {eer_svm:.4f}")
    print("\n")
    print("Random Forest Report:\n", rf_report)
    print(f"Random Forest Max FRR: {max_frr_rf:.4f}, Min FRR: {min_frr_rf:.4f}, Avg FRR: {avg_frr_rf:.4f}")
    print(f"Random Forest Max FAR: {max_far_rf:.4f}, Min FAR: {min_frr_rf:.4f}, Avg FAR: {avg_far_rf:.4f}")
    print(f"Random Forest Equal Error Rate (EER): {eer_rf:.4f}")
    print("\n")
    summary_table = [
        ["KNN", knn_accuracy, max_frr_knn, min_frr_knn, avg_frr_knn, max_far_knn, min_far_knn, avg_far_knn, eer_knn],
        ["SVM", svm_accuracy, max_frr_svm, min_frr_svm, avg_frr_svm, max_far_svm, min_far_svm, avg_far_svm, eer_svm],
        ["Random Forest", rf_accuracy, max_frr_rf, min_frr_rf, avg_frr_rf, max_far_rf, min_far_rf, avg_far_rf, eer_rf]
    ]
    headers = ["Method", "Accuracy", "Max FRR", "Min FRR", "Avg FRR", "Max FAR", "Min FAR", "Avg FAR", "EER"]
    print(tabulate(summary_table, headers, tablefmt="grid"))


if __name__ == '__main__':
    main()
