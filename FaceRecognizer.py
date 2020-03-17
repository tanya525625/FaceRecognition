import cv2
import numpy as np
from matplotlib import pyplot as plt


class FaceRecognizer:
    def __init__(self, feature_extraction_method, method_args=None):
        self.feature_extraction_method = feature_extraction_method
        self.method_args = method_args

    def fit(self, X_train, y_train):
        X_train = _prepare_dataset(X_train, self.feature_extraction_method, self.method_args)
        # make_hist_plot(X_train[0])

    def predict(self, X_test):
        pass


def _prepare_dataset(images, feature_extraction_method, method_args):
    hists = []
    for image in images:
        method_args.update({'images': [image]})
        hist = feature_extraction_method(**method_args)
        hists.append(hist)

    print(type(hists[0]))
    return hists


def make_hist_plot(hist):
    x = np.arange(256)
    plt.plot(x, hist)
    plt.show()
