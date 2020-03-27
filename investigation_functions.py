import os

import cv2
import yaml
import numpy as np
from matplotlib import pyplot as plt

from FaceRecognizer import FaceRecognizer, VoteRecognizer


def person_recognition_example(db, proportion, feature_extraction_methods, args, images_path):
    X_train, y_train, X_test, y_test = train_and_test_split(db, proportion)
    [img_width, img_height] = X_test[0].shape
    new_width, new_height = img_width * 3, img_height * 3
    resized_img = cv2.resize(X_test[0], (new_width, new_height))
    show_image(resized_img, "Original image")
    cv2.imwrite(os.path.join(images_path, "Original image.png"), X_test[0])

    for feat_extr_meth, par in zip(feature_extraction_methods, args):
        recognizer = FaceRecognizer(feat_extr_meth, par)
        recognizer.fit(X_train, y_train)
        prediction, pred_number = recognizer.predict(X_test[0])
        method_name = feat_extr_meth.__name__
        resized_img = cv2.resize(X_train[pred_number], (new_width, new_height))
        show_image(resized_img, method_name)
        cv2.imwrite(os.path.join(images_path, f"{method_name}.png"), X_train[pred_number])


def train_and_test_split(dataset, proportion):
    X_train, y_train, X_test, y_test = [], [], [], []

    for i, person in enumerate(dataset.keys()):
        train_values_count = int(len(dataset[person]) * proportion)
        X_train.extend(dataset[person][:train_values_count])
        y_train.extend([i] * train_values_count)
        X_test.extend(dataset[person][train_values_count:])
        y_test.extend([i] * (len(dataset[person]) - train_values_count))

    return X_train, y_train, X_test, y_test


def make_plot(x, y, file_path, method_name, investigation_type):
    fig, ax = plt.subplots()
    ax.set_xlabel("parameter")
    ax.set_ylabel("accuracy")
    ax.title.set_text(f"{investigation_type}: {method_name}")
    ax.plot(x, y)
    ax.grid()
    fig.savefig(file_path)
    fig.clf()


def investigate_params(X_train, y_train, X_test, y_test,
                       feature_extraction_methods, args,
                       param_graphs_path, config_path):
    best_params = []
    config_dict = {}

    for feat_extr_meth, par in zip(feature_extraction_methods, args):
        accuracy_values = []
        for arg in list(par.values())[0]:
            arg_dict = {list(par.keys())[0]: arg}
            if par["const_args"] is not None:
                arg_dict.update(par["const_args"])
            recognizer = FaceRecognizer(feat_extr_meth, arg_dict)
            recognizer.fit(X_train, y_train)
            accuracy = find_accuracy(X_test, y_test, recognizer)

            accuracy_values.append(accuracy)

        best_param = list(par.values())[0][np.argmax(accuracy_values)]
        best_params.append(best_param)
        method_name = feat_extr_meth.__name__
        method_dict = {
            method_name: {
                'best_param': best_param,
                'best_accuracy': max(accuracy_values)
            }
        }
        config_dict.update(method_dict)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        paameters = list(par.values())[0]
        file_path = os.path.join(param_graphs_path, method_name)
        make_plot(paameters, accuracy_values, file_path, method_name, "Parameters investigation")

        print(f"Parameters investigation for {method_name} is done")

    return best_params


def investigate_dataset_size(db, feature_extraction_methods, args, proportions,
                             best_params, config_path, size_invest_path):
    config_dict = {}

    for feat_extr_meth, par, best_param in zip(feature_extraction_methods, args, best_params):
        arg_dict = {list(par.keys())[0]: best_param}
        if par["const_args"] is not None:
            arg_dict.update(par["const_args"])

        accuracy_values = []
        for proportion in proportions:
            X_train, y_train, X_test, y_test = train_and_test_split(db, proportion)
            recognizer = FaceRecognizer(feat_extr_meth, arg_dict)
            recognizer.fit(X_train, y_train)
            accuracy = find_accuracy(X_test, y_test, recognizer)
            accuracy_values.append(accuracy)

        method_name = feat_extr_meth.__name__
        method_dict = {
            method_name: {
                'best_param': proportions[np.argmax(accuracy_values)],
                'best_accuracy': max(accuracy_values)
            }
        }
        config_dict.update(method_dict)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        train_imgs_count = list(map(lambda x: int(x * 10), proportions))
        file_path = os.path.join(size_invest_path, method_name)
        make_plot(train_imgs_count, accuracy_values, file_path, method_name, "Dataset size investigation")
        print(f"Dataset size investigation for {method_name} is done")


def find_accuracy(X_test, y_test, recognizer):
    right_predictions_count = 0
    all_values_count = len(y_test)
    for test_img, test_person in zip(X_test, y_test):
        prediction = recognizer.predict(test_img)
        if prediction[0] == test_person:
            right_predictions_count += 1

    return right_predictions_count / all_values_count * 100


def show_image(img, title):
    cv2.imshow(title, img)
    cv2.namedWindow(title)
    cv2.waitKey()


def voter_investigation(db, feature_extraction_methods, args, best_args, proportions,
                        best_params, config_path, graph_path, accuracies):
    config_dict = {}
    fig, ax = plt.subplots()
    train_imgs_count = list(map(lambda x: int(x * 10), proportions))
    #
    # for feat_extr_meth, par, best_param in zip(feature_extraction_methods, args, best_params):
    #     arg_dict = {list(par.keys())[0]: best_param}
    #     if par["const_args"] is not None:
    #         arg_dict.update(par["const_args"])
    #
    #     accuracy_values = []
    #     for proportion in proportions:
    #         X_train, y_train, X_test, y_test = train_and_test_split(db, proportion)
    #         recognizer = FaceRecognizer(feat_extr_meth, arg_dict)
    #         recognizer.fit(X_train, y_train)
    #         accuracy = find_accuracy(X_test, y_test, recognizer)
    #         accuracy_values.append(accuracy)
    #
    #     method_name = feat_extr_meth.__name__
    #     method_dict = {
    #         method_name: {
    #             'best_param': proportions[np.argmax(accuracy_values)],
    #             'best_accuracy': max(accuracy_values)
    #         }
    #     }
    #     config_dict.update(method_dict)
    #     with open(config_path, 'w') as f:
    #         yaml.dump(config_dict, f)

        # fig = make_plot_for_voting_comparison(train_imgs_count, accuracy_values,
        #                                       method_name, "Comparison with voting method", fig, ax)
        # print(f"Dataset size investigation for {method_name} is done")

    voting_accuracies = []
    for proportion in proportions:
        voting_recognizer = VoteRecognizer(feature_extraction_methods, accuracies, best_args)
        X_train, y_train, X_test, y_test = train_and_test_split(db, proportion)
        voting_recognizer.fit(X_train, y_train)
        voting_accuracies.append(find_accuracy(X_test, y_test, voting_recognizer))
    print(f"Dataset size investigation for voting method is done")
    method_dict = {
        "Voter": {
            'best_param': proportions[np.argmax(voting_accuracies)],
            'best_accuracy': max(voting_accuracies)
        }
    }
    config_dict.update(method_dict)
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)

    fig = make_plot_for_voting_comparison(train_imgs_count, voting_accuracies,
                                          "voting method", "Voting method", fig, ax)

    # ax.legend(["Hist", "DFT", "DCT", "Sc-Scale", "Sliding window", "Voting method"])
    graph_path = os.path.join(graph_path, "voting_graphs.png")
    fig.savefig(graph_path)


def make_plot_for_voting_comparison(x, y, method_name, investigation_type, fig, ax):
    ax.set_xlabel("parameter")
    ax.set_ylabel("accuracy")
    ax.title.set_text(f"{investigation_type}")
    ax.plot(x, y, label=method_name)
    ax.grid()

    return fig
