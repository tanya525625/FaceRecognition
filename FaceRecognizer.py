import numpy as np
from matplotlib import pyplot as plt


class FaceRecognizer:
    def __init__(self, feature_extraction_method, prepr_function, method_args=None):
        self.feature_extraction_method = feature_extraction_method
        self.method_args = method_args
        self.prepr_function = prepr_function
        self.X_train, self.y_train = [], []

    def fit(self, X_train, y_train):

        self.X_train = _prepare_dataset(X_train, self.feature_extraction_method, self.method_args, self.prepr_function)
        self.y_train = y_train
        # make_hist_plot(X_train[0])

    def predict(self, img):
        img_test = _prepare_dataset([img], self.feature_extraction_method, self.method_args, self.prepr_function)
        distances = [_find_vectors_distance(img_test, train_img) for train_img in self.X_train]
        #print(distances)
        return self.y_train[np.argmin(distances)]


def _find_vectors_distance(x, y):
    # print(x)
    # print(y)
    return np.linalg.norm(np.array(x) - np.array(y))


def _prepare_dataset(images, feature_extraction_method, method_args, prepr_function):
    hists = []
    for image in images:
        hist = feature_extraction_method(prepr_function(image), **method_args)
        hists.append(hist)

    return hists


def make_hist_plot(hist):
    x = np.arange(256)
    plt.plot(x, hist)
    plt.show()
