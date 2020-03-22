import os
import random

import cv2
import numpy as np
from scipy.fftpack import dct
import pywt
from matplotlib import pyplot as plt

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
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        # print(test_img)
        # print(recognizer.X_train)
        prediction = recognizer.predict(test_img)

        if prediction == test_person:
            right_predictions_count += 1

    return right_predictions_count/all_values_count * 100


def find_point_on_line(x, k, x_0=0):
    return x_0 + k*x


def generate_random_points(img_shape, poins_count, epsilon=40):
    coord_x = []
    coord_y = []
    x0_values = [0, 50, 0, 0]
    k_values = [1, -1, 0.1, 10]
    for i in range(poins_count):
        x = random.randint(0, img_shape[0] - 1)
        for k, x0 in zip(k_values, x0_values):
            y = find_point_on_line(x, k, x0)
            if y - epsilon < 0 or y + epsilon < 0:
                continue
            else:
                y = random.randint(int(y - epsilon), int(y + epsilon))
                if y < img_shape[1]:
                    coord_y.append(y)
                    coord_x.append(x)

    return list(zip(coord_x, coord_y))


def random_method(img, coordinate_points):
    return [img[point[0]][point[1]][color] for point in coordinate_points for color in range(3)]


if __name__ == "__main__":
    path_to_db = "orl_faces"
    param_graphs_path = "parameters_investigation_graphs"
    size_invest_path = "dataset_size_investigation"

    db = make_db(path_to_db)

    proportion = 0.8
    X_train, y_train, X_test, y_test = train_and_test_split(db, proportion)
    # img_shape = np.array(X_train[0]).shape
    # points_count = 300
    # random_coordinates = generate_random_points(img_shape, points_count)

    hist_par_args = {}
    hist_args = {
        'histSize': [[x] for x in range(10, 257, 20)],
        "const_args": {
            'channels': [0],
            'mask': None,
            'ranges': [0, 256]
        }
    }
    dft_args = {
        's': [[x] for x in range(10, 200, 50)],
        "const_args": None
    }

    # random_args = {'coordinate_points': random_coordinates}

    dct_args = {
        'n': range(1, 15),
        "const_args": None
    }
    #scaler_args = {}
    dwt_args = {
        'wavelet': ['bior1.3', 'db1'],
        "const_args": None
    }
    # gradient_args = {
    #     'ddepth': cv2.CV_64F,
    #     'dx': 1,
    #     'dy': 0
    # }

    window_args = {
        'window_size': range(1, 21, 1),
        "const_args": None
    }

    scaler_args = {
        'area': range(1, 21, 1),
        "const_args": None
    }

    feature_extraction_methods = [cv2.calcHist, np.fft.fft2, dct, pywt.dwt, ipf.sliding_window_method]
    feature_extraction_methods = [ipf.scaler_method]
    preprocessing_methods = [ipf.hist_preprocessing, ipf.dct_preprocessing, ipf.scaler_preprocessing,
                             ipf.scaler_preprocessing, ipf.scaler_preprocessing]
    preprocessing_methods = [ipf.scaler_preprocessing]
    args = [hist_args, dft_args, dct_args, dwt_args, window_args]
    args = [scaler_args]

    for feat_extr_meth, prepr_meth, par in zip(feature_extraction_methods, preprocessing_methods, args):
        accuracy_values = []
        for arg in list(par.values())[0]:
            arg_dict = {list(par.keys())[0]: arg}
            if par["const_args"] is not None:
                arg_dict.update(par["const_args"])
            recognizer = FaceRecognizer(feat_extr_meth, prepr_meth, arg_dict)
            recognizer.fit(X_train, y_train)
            #print(recognizer.X_train)
            # print(feat_extr_meth.__name__)
            accuracy = find_accuracy(X_test, y_test, recognizer)

            accuracy_values.append(accuracy)

        best_param = list(par.values())[0][np.argmax(accuracy_values)]
        print(max(accuracy_values))
        print(best_param)
        plt.plot(list(par.values())[0], accuracy_values)
        file_path = os.path.join(param_graphs_path, feat_extr_meth.__name__)
        plt.grid()
        plt.savefig(file_path)
        plt.clf()

    best_param = 7
    proportions = [0.2, 0.4, 0.6, 0.8]
    accuracy_values = []

    for feat_extr_meth, prepr_meth, par in zip(feature_extraction_methods, preprocessing_methods, args):
        arg_dict = {list(par.keys())[0]: best_param}
        if par["const_args"] is not None:
            arg_dict.update(par["const_args"])
        for proportion in proportions:
            X_train, y_train, X_test, y_test = train_and_test_split(db, proportion)
            recognizer = FaceRecognizer(feat_extr_meth, prepr_meth, arg_dict)
            recognizer.fit(X_train, y_train)
            #print(recognizer.X_train)
            accuracy = find_accuracy(X_test, y_test, recognizer)
            accuracy_values.append(accuracy)

        best_value = max(accuracy_values)
        print(best_value)
        train_imgs_count = list(map(lambda x: int(x * 10), proportions))
        plt.plot(train_imgs_count, accuracy_values)
        file_path = os.path.join(size_invest_path, feat_extr_meth.__name__)
        plt.grid()
        plt.savefig(file_path)
        plt.clf()




