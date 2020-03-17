import os

import cv2
import numpy as np

from FaceRecognizer import FaceRecognizer


def detect_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(os.path.join(cv2.haarcascades, "haarcascade_frontalface_default.xml"))
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    return gray_img[y:y + w, x:x + h], faces[0]


def make_db(path_to_dataset):
    people = {}

    for images_dir in os.listdir(path_to_dataset):
        if images_dir.startswith("s"):
            dir_paths = os.path.join(path_to_dataset, images_dir)
            faces = []
            for image_path_name in os.listdir(dir_paths):
                image_path = os.path.join(dir_paths, image_path_name)
                image = cv2.imread(image_path)
                face, rect = detect_face(image)
                if face is not None:
                    faces.append(face)
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

    recognizer = FaceRecognizer(cv2.calcHist, hist_args)
    recognizer.fit(X_train, y_train)

