import numpy as np

from sklearn.preprocessing import StandardScaler
from FaceRecognizer import _find_vectors_distance


def hist_preprocessing(img):
    return [img]


def dft_preprocessing(img):
    return np.float32(img)


def dct_preprocessing(img):
    return img[0]


def scaler_preprocessing(img):
    return img


def scaler(img):
    scalers = {}
    for i in range(img.shape[1]):
        scalers[i] = StandardScaler()
        img[:, i, :] = scalers[i].fit_transform(img[:, i, :])

    return img


def find_gradient_distance(row_1, row_2):
    return abs(np.gradient(row_1) - np.gradient(row_2))


def sliding_window_method(img, window_size):
    vectors = []
    img_shape = img.shape
    for row_ind in range(window_size, img_shape[0]-window_size):
        vectors.append(_find_vectors_distance(img[row_ind - window_size], img[row_ind + window_size]))

    return vectors
