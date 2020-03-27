import numpy as np


class FaceRecognizer:
    def __init__(self, feature_extraction_method,  method_args=None):
        self.feature_extraction_method = feature_extraction_method
        self.method_args = method_args
        self.X_train, self.y_train = [], []

    def fit(self, X_train, y_train):
        self.X_train = _prepare_dataset(X_train, self.feature_extraction_method, self.method_args)
        self.y_train = y_train

    def predict(self, img):
        img_test = _prepare_dataset([img], self.feature_extraction_method, self.method_args)
        distances = [_find_vectors_distance(img_test, train_img) for train_img in self.X_train]
        return self.y_train[np.argmin(distances)], np.argmin(distances)


def _find_vectors_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


def _prepare_dataset(images, feature_extraction_method, method_args):
    vectors = []
    for image in images:
        vector = feature_extraction_method(image, **method_args)
        vectors.append(vector)

    return vectors


class VoteRecognizer:
    def __init__(self, extraction_methods: list, accuracies: list,
                 methods_args: list = None):
        self.extraction_methods = extraction_methods
        self.methods_args = methods_args
        self.accuracies = accuracies
        self.recognizers = []
        self.classes = []
        # self.X_train, self.y_train = [], []

    def fit(self, X_train, y_train):
        for method, args, accuracy in zip(self.extraction_methods, self.methods_args, self.accuracies):
            recognizer = FaceRecognizer(method, args)
            recognizer.fit(X_train, y_train)
            self.recognizers.append(recognizer)
            if method == self.extraction_methods[-1]:
                self.classes = set(recognizer.y_train)
            # self.X_train.append(recognizer.X_train)
            # self.y_train.append(recognizer.y_train)

    def predict(self, img):
        predictions = np.zeros(len(self.classes)+1)
        for recognizer, accuracy in zip(self.recognizers, self.accuracies):
            prediction = recognizer.predict(img)[0]
            #print("Predict by meth", prediction)
            predictions[prediction] += accuracy
        #print("Prediction by vote", np.argmax(predictions))
        return np.argmax(predictions), None
