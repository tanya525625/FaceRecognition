import numpy as np
import cv2

from FaceRecognizer import _find_vectors_distance


def hist(img, histSize, channels, mask, ranges):
    return cv2.calcHist([img], histSize=histSize, channels=channels, mask=mask, ranges=ranges)


def sliding_window(img, window_size):
    vectors = []
    img_shape = img.shape
    for row_ind in range(window_size, img_shape[0]-window_size):
        vectors.append(_find_vectors_distance(img[row_ind - window_size], img[row_ind + window_size]))

    return vectors


def scaler(img, area):
    vectors = []
    for row_ind, col_ind in zip(range(0, img.shape[0] - area, area), range(0, img.shape[1] - area, area)):
        values = []
        for offset in range(area):
            values.append(img[row_ind + offset][col_ind + offset])
        vectors.append(np.mean(values))

    return vectors


def dft(image, p):
    s = [p, p]
    image = np.fft.fft2(image, s=s)

    return image


def find_gradient_distance(row_1, row_2):
    return abs(np.gradient(row_1) - np.gradient(row_2))
