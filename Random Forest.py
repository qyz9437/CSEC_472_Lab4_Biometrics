import numpy as np
import cv2
import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from sklearn.ensemble import RandomForestClassifier
import random

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    equalized = cv2.equalizeHist(img)
    return cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def detect_features(path, disp=False):
    img = preprocess_image(path)
    dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    keypoints = cv2.goodFeaturesToTrack(dst, maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=3, useHarrisDetector=True)

    if disp:
        img2 = img.copy()
        for i in keypoints:
            x, y = i.ravel()
            cv2.circle(img2, (x, y), 3, 255, -1)
        plt.imshow(img2)
        plt.show()

    return keypoints

def extract_features(keypoints, fixed_size=500):
    features = np.array([m.ravel() for m in keypoints if m is not None]).flatten()
    if len(features) > fixed_size:
        features = features[:fixed_size]
    elif len(features) < fixed_size:
        features = np.pad(features, (0, fixed_size - len(features)), 'constant')
    return features

def train_random_forest(train_features, train_labels):
    classifier = RandomForestClassifier(n_estimators=100, max_depth=200, min_samples_split=10, random_state=42)
    classifier.fit(train_features, train_labels)
    return classifier

def evaluate_model(classifier, test_features, test_labels):
    frr = 0
    far = 0

    r = random.sample(range(500), 100)
    random_test_features = []
    random_test_labels = []
    for i in r:
        random_test_features.append(test_features[i])
        random_test_labels.append(test_labels[i])
    predictions = (classifier.predict_proba(random_test_features)[:, 1] >= .28).astype(int)
    accuracy = accuracy_score(random_test_labels, predictions)
    report = classification_report(random_test_labels, predictions, zero_division=0)

    sum_frr = 0
    sum_far = 0
    true_rejects = 0
    true_accepts = 0

    for j in range(len(random_test_labels)):
        if (random_test_labels[j] == 1):
            true_accepts += 1
        else:
            true_rejects += 1
        if random_test_labels[j] == 1 and predictions[j] == 0:
            sum_frr += 1
        if random_test_labels[j] == 0 and predictions[j] == 1:
            sum_far += 1

    frr = sum_frr / true_accepts
    far = sum_far / true_rejects

    return accuracy, report, frr, far

def compare_fingerprints(classifier, path_a, path_b, similarity_threshold=0.28, debug=False):
    keypoints_a = detect_features(path_a, debug)
    keypoints_b = detect_features(path_b, debug)

    features_a = extract_features(keypoints_a)
    features_b = extract_features(keypoints_b)

    combined_features = np.concatenate((features_a, features_b))
    similarity = (classifier.predict_proba(combined_features)[:, 1] >= similarity_threshold).astype(int)

    return True if similarity[0] == 1 else False

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
            ref_keypoints = detect_features(ref_path, disp=False)
            subj_keypoints = detect_features(subj_path, disp=False)

            ref_features = extract_features(ref_keypoints)
            subj_features = extract_features(subj_keypoints)
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
            subj_keypoints = detect_features(subj_path, disp=False)
            subj_features = extract_features(subj_keypoints)
            combined_features = np.concatenate((ref_features, subj_features))

            paired_features.append(combined_features)
            labels.append(0)
        file_id = os.path.basename(dif_path_2).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            subj_keypoints = detect_features(subj_path, disp=False)
            subj_features = extract_features(subj_keypoints)
            combined_features = np.concatenate((ref_features, subj_features))

            paired_features.append(combined_features)
            labels.append(0)
        file_id = os.path.basename(dif_path_3).split('_')[0][1:]
        subj_path = subject_image_paths.get(file_id)
        if subj_path:
            subj_keypoints = detect_features(subj_path, disp=False)
            subj_features = extract_features(subj_keypoints)
            combined_features = np.concatenate((ref_features, subj_features))

            paired_features.append(combined_features)
            labels.append(0)

    train_features = np.array(paired_features)
    train_labels = np.array(labels, dtype=int)

    test_features = []
    test_labels = []

    for i in range(500):
        ref_path = reference_image_paths[i + 1500]
        ref_keypoints = detect_features(ref_path, disp=False)
        ref_features = extract_features(ref_keypoints)
        k = random.randint(0, 1)
        if k == 0:
            file_id = os.path.basename(ref_path).split('_')[0][1:]
            subj_path = subject_image_paths.get(file_id)
            if subj_path:
                subj_keypoints = detect_features(subj_path, disp=False)
                subj_features = extract_features(subj_keypoints)
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
                subj_keypoints = detect_features(subj_path, disp=False)
                subj_features = extract_features(subj_keypoints)
                combined_features = np.concatenate((ref_features, subj_features))
                test_features.append(combined_features)
                test_labels.append(0)

    max_frr = 0
    min_frr = 1
    max_far = 0
    min_far = 1
    sum_frr = 0
    sum_far = 0

    rf_classifier = train_random_forest(train_features, train_labels)

    rf_accuracy, rf_report, frr, far = evaluate_model(
        rf_classifier, test_features, test_labels)

    print("Random Forest Accuracy: ", rf_accuracy)
    print("Random Forest Report:\n", rf_report)
    print(f"Random Forest FRR:  {frr:.4f}")
    print(f"Random Forest FAR: {far:.4f}")
    print("\n")

    summary_table = [
        ["Random Forest", rf_accuracy, frr, far]
    ]
    headers = ["Method", "Accuracy", "FRR", "FAR"]
    print(tabulate(summary_table, headers, tablefmt="grid"))

if __name__ == '__main__':
    main()
