import numpy as np
import cv2
import glob

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
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
    keypoints = cv2.goodFeaturesToTrack(harris_response, maxCorners=150, qualityLevel=0.01, minDistance=10, blockSize=3, useHarrisDetector=True)

    if display:
        image_copy = processed_image.copy()
        for keypoint in keypoints:
            x, y = keypoint.ravel()
            cv2.circle(image_copy, (x, y), 3, 255, -1)
        plt.imshow(image_copy)
        plt.show()

    return keypoints

def extract_features(keypoints, fixed_size=500):
    features = np.array([keypoint.ravel() for keypoint in keypoints if keypoint is not None]).flatten()
    if len(features) > fixed_size:
        features = features[:fixed_size]
    elif len(features) < fixed_size:
        features = np.pad(features, (0, fixed_size - len(features)), 'constant')
    return features

def train_svm(train_features, train_labels):
    classifier = SVC(gamma='scale', kernel='rbf', probability=True)
    classifier.fit(train_features, train_labels)
    return classifier

def evaluate_svm(classifier, test_features, test_labels):
    frr = 0
    far = 0

    random_indices = random.sample(range(500), 100)
    random_test_features = [test_features[i] for i in random_indices]
    random_test_labels = [test_labels[i] for i in random_indices]
    probabilities = classifier.predict_proba(random_test_features)[:, 1]
    predictions = (probabilities >= 0.13).astype(int)
    accuracy = accuracy_score(random_test_labels, predictions)
    report = classification_report(random_test_labels, predictions, zero_division=0)

    true_accepts = 0
    true_rejects = 0
    sum_frr = 0
    sum_far = 0

    for i in range(len(random_test_labels)):
        if random_test_labels[i] == 1:
            true_accepts += 1
        else:
            true_rejects += 1
        if random_test_labels[i] == 1 and predictions[i] == 0:
            sum_frr += 1
        if random_test_labels[i] == 0 and predictions[i] == 1:
            sum_far += 1

    frr = sum_frr / true_accepts
    far = sum_far / true_rejects

    return accuracy, report, frr, far

def compare_fingerprints(classifier, image_a, image_b, similarity_threshold=0.13, debug=False):
    minutiae_a = detect_minutiae(image_a, debug)
    minutiae_b = detect_minutiae(image_b, debug)

    features_a = extract_features(minutiae_a)
    features_b = extract_features(minutiae_b)

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

    svm_model = train_svm(train_features, train_labels)

    svm_accuracy, svm_report, frr, far = evaluate_svm(
        svm_model, test_features, test_labels)

    print("SVM Accuracy: ", svm_accuracy)
    print("SVM Report:\n", svm_report)
    print(f"SVM  FRR: {frr:.4f}")
    print(f"SVM FAR: {far:.4f}")
    print("\n")

    summary_table = [
        ["SVM", svm_accuracy, frr, far]
    ]
    headers = ["Method", "Accuracy", "FRR", "FAR"]
    print(tabulate(summary_table, headers, tablefmt="grid"))

if __name__ == '__main__':
    main()
