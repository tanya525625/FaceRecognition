import os

import cv2
from scipy.fftpack import dct

from tools import detect_face
import image_preprocessing_functions as ipf
import investigation_functions as inv_func


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
            person = f'person_{images_dir.replace("s", "")}'
            people.update({person: faces})

    return people


if __name__ == "__main__":
    path_to_db = "orl_faces"
    size_invest_path = os.path.join("graphs", "dataset_size_investigation")
    param_graphs_path = os.path.join("graphs", "parameters_investigation_graphs")
    voter_inv_path = os.path.join("graphs", "voter_graphs")
    params_inv_config_path = os.path.join("configs", "best_params.yml")
    dataset_inv_config_path = os.path.join("configs", "dataset_size_investigation.yml")
    voter_inv_config_path = os.path.join("configs", "voter_investigation.yml")

    investigation_mode = False
    voter_investigation_mode = True
    feature_extraction_methods = [ipf.hist, ipf.dft, dct,
                                  ipf.scaler, ipf.sliding_window]

    db = make_db(path_to_db)
    proportions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    hist_args = {
        'histSize': [[x] for x in range(10, 257, 20)],
        "const_args": {
            'channels': [0],
            'mask': None,
            'ranges': [0, 256]
        }
    }

    dct_args = {
        'n': range(10, 210, 10),
        "const_args": None
    }

    window_args = {
        'window_size': range(1, 21, 1),
        "const_args": None
    }

    scaler_args = {
        'area': range(1, 21, 1),
        "const_args": None
    }

    dft_args = {
        'p': range(50, 200, 10),
        "const_args": None
    }
    args = [hist_args, dft_args, dct_args, scaler_args, window_args]

    if investigation_mode:
        proportion = 0.6
        X_train, y_train, X_test, y_test = inv_func.train_and_test_split(db, proportion)

        best_params = inv_func.investigate_params(X_train, y_train, X_test, y_test,
                                                  feature_extraction_methods, args,
                                                  param_graphs_path, params_inv_config_path)

        inv_func.investigate_dataset_size(db, feature_extraction_methods, args, proportions,
                                          best_params, dataset_inv_config_path, size_invest_path)
        print(best_params)
    else:
        best_hist_args = {
            'histSize': [30],
            'channels': [0],
            'mask': None,
            'ranges': [0, 256]
        }
        best_dft_args = {'p': 120}
        best_dct_args = {'n': 90}
        best_scaler_args = {'area': 7}
        best_window_args = {'window_size': 2}
        best_args = [best_hist_args, best_dft_args, best_dct_args, best_scaler_args, best_window_args]
        if voter_investigation_mode:
            accuracies = [1, 0.45, 0.45, 0.4, 0.8]
            best_params = [[30], 120, 90, 7, 2]
            inv_func.voter_investigation(db, feature_extraction_methods, args, best_args, proportions,
                                         best_params, voter_inv_config_path, voter_inv_path, accuracies)
        else:
            proportion = 0.8
            images_path = "predictions"

            inv_func.person_recognition_example(db, proportion, feature_extraction_methods, best_args, images_path)
