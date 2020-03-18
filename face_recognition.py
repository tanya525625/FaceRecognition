import os

import cv2
import numpy as np
from scipy.fftpack import dct
from sklearn.preprocessing import StandardScaler
import pywt
from scipy.signal import cwt, ricker

from FaceRecognizer import FaceRecognizer
from tools import detect_face
import image_preprocessing_functions as ipf


def make_db(path_to_dataset, is_detection=False):
    people = {}

    for images_dir in os.listdir(path_to_dataset):
        if images_dir.startswith("s"):
            dir_paths = os.path.join(path_to_dataset, images_dir)
            faces = []
            for image_path_name in os.listdir(dir_paths):
                image_path = os.path.join(dir_paths, image_path_name)
                image = cv2.imread(image_path)
                if is_detection:
                    face, rect = detect_face(image)
                    if face is not None:
                        faces.append(face)
                else:
                    faces.append(image)
                    # cv2.imshow("Training on image...", face)
                    # cv2.waitKey(100)
            person = f'person_{images_dir.replace("s", "")}'
            people.update({person: faces})

    return people


def train_and_test_split(dataset, proportion):
    X_train, y_train, X_test, y_test = [], [], [], []

    for i, person in enumerate(dataset.keys()):
        train_values_count = int(len(dataset[person]) * proportion)
        X_train.extend(dataset[person][:train_values_count])
        y_train.extend([i] * train_values_count)
        X_test.extend(dataset[person][train_values_count:])
        y_test.extend([i] * (len(dataset[person]) - train_values_count))

    return X_train, y_train, X_test, y_test


def find_accuracy(X_test, y_test, recognizer):
    right_predictions_count = 0
    all_values_count = len(y_test)
    for test_img, test_person in zip(X_test, y_test):
        prediction = recognizer.predict(test_img)
        if prediction == test_person:
            right_predictions_count += 1

    return right_predictions_count/all_values_count * 100


if __name__ == "__main__":
    path_to_db = "orl_faces"

    db = make_db(path_to_db)
    proportion = 0.6
    X_train, y_train, X_test, y_test = train_and_test_split(db, proportion)
    hist_args = {
        'channels': [0],
        'mask': None,
        'histSize': [256],
        'ranges': [0, 256]
    }

    dft_args = {
        'norm': 'ortho'
    }

    dct_args = {}
    scaler_args = {}
    widths = np.arange(1, 31)
    dwt_args = {
        'wavelet': 'bior1.3'
    }
    gradient_args = {
        'ddepth': cv2.CV_64F,
        'dx': 1,
        'dy': 0
    }


    feature_extraction_methods = [cv2.calcHist, np.fft.fft, dct, pywt.dwt, cv2.Sobel]
    preprocessing_methods = [ipf.hist_preprocessing, ipf.dct_preprocessing, ipf.scaler_preprocessing]
    recognizer = FaceRecognizer(pywt.dwt, ipf.scaler_preprocessing, dwt_args)
    recognizer.fit(X_train, y_train)
    accuracy = find_accuracy(X_test, y_test, recognizer)
    print(accuracy)
