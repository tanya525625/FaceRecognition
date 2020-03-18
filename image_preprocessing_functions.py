import cv2
import numpy as np

from sklearn.preprocessing import StandardScaler


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