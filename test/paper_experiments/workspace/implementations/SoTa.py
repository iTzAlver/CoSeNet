# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
"""
File info:
"""
# Import statements:
import logging
import json
import os
import time
import tensorflow as tf
import numpy as np
from tensorflow import keras
from basenet import BaseNetDatabase, window_diff
DB_PATH = '../data/wk/wikipedia_dataset_256.db'
DB_PATH_SYNTH = '../data/sym/32k_32t_02w.db'
DATA_PATH = '../data/baseline_comparison.json'
DATA_PATH_SYNTH = '../data/baseline_comparison_synth.json'
PAR_PATH = '../data/baseline_comparison_parameters.json'


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        MAIN CLASS                         #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
class Prepare:
    def __init__(self, test_batch: int = 346, data_path: str = DB_PATH, synth=False):
        # Load database:
        logging.info(f'[+] <Prepare>: Pre-processing database.')
        if not synth:
            db = BaseNetDatabase.load(data_path)
        else:
            if data_path == DB_PATH:
                db = BaseNetDatabase.load(DB_PATH_SYNTH)
            else:
                db = BaseNetDatabase.load(data_path)
        # Extract database:
        matrices = np.squeeze(np.concatenate([db.xtrain, db.xval]), axis=-1)
        labels = np.concatenate([db.ytrain, db.yval])
        # Compute divisions:
        n_divs = int(matrices.shape[0] / test_batch)
        # Compute batches:
        matrix_batch = [matrices[i * test_batch:(i + 1) * test_batch] for i in range(n_divs)]
        labels_batch = [labels[i * test_batch:(i + 1) * test_batch] for i in range(n_divs)]
        sizes_batch = np.random.randint(50, 255, size=n_divs)
        # Get batches:
        self.matrix = [m[:, :s, :s] for m, s in zip(matrix_batch, sizes_batch)]
        self.labels = [lbl[:, :s] for lbl, s in zip(labels_batch, sizes_batch)]
        logging.info(f'[-] <Prepare>: Database pre-processed.')


class Predict:
    def __init__(self, model, matrix: list[np.ndarray], pred_labels: list[np.ndarray], parameters: tuple = (),
                 synth=False):
        logging.info(f'[+] <Predict>: Predicting outcomes for model {model.__name__}.')
        if not synth:
            data_path = DATA_PATH
        else:
            data_path = DATA_PATH_SYNTH
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                results = json.load(f)
            logging.info(f'[#] <Predict>: Results loaded from {data_path}.')
        else:
            results = dict()
            logging.info(f'[!] <Predict>: Results file not found, creating new one.')
        __res: (None, np.ndarray) = None
        results_list = list()
        for mat, labels in zip(matrix, pred_labels):
            logging.info(f'[#] <Predict>: Computing size of {mat.shape[-1]}.')
            tik = time.perf_counter()
            __res = model(mat, *parameters)
            tik_tok = time.perf_counter() - tik
            segmentation = self.__label2seg(__res)
            wd = window_diff(tf.convert_to_tensor(segmentation), tf.convert_to_tensor(labels))
            mse = tf.keras.losses.MeanSquaredError()(segmentation, labels)
            mae = tf.keras.losses.MeanAbsoluteError()(segmentation, labels)
            this_results = {'window_diff': float(wd), 'mse': float(mse), 'mae': float(mae),
                            'size': int(segmentation.shape[-1]), 'time': tik_tok}
            results_list.append(this_results)
            logging.info(f'[#] <Results>: Results:\n{this_results}')
        results[model.__name__] = results_list  # List of results for the model.
        with open(data_path, 'w') as f:
            json.dump(results, f, indent=4)
            logging.info(f'[#] <Predict>: Results saved to {data_path}.')
        logging.info(f'[-] <Predict>: Outcomes predicted for model {model.__name__}.')

    @staticmethod
    def __label2seg(labels: list[np.ndarray]) -> np.ndarray:
        segs = list()
        for label in labels:
            seg = np.zeros(len(label))
            last_label = -1
            for index, label_number in enumerate(label):
                seg[index] = 1 if label_number != last_label else 0
                last_label = label_number
            segs.append(seg)
        return np.array(segs)


class Fit:
    def __init__(self, model, matrix: np.ndarray, pred_labels: np.ndarray, parameters: list, epochs: int = 10):
        logging.info(f'[+] <Fit>: Fitting outcomes for model {model.__name__}.')
        self.best_parameters = ()
        if 'deep' not in model.__name__.lower():
            self.hc_local_search(model, matrix, pred_labels, parameters, epochs)
        else:
            self.bp_fit(model, matrix, pred_labels)
        with open(PAR_PATH, 'w') as f:
            try:
                rex = json.load(f)
            except Exception as ex:
                logging.info(f'\t[!] <Fit>: {ex}')
                rex = dict()
            rex[model.__name__] = self.best_parameters
            json.dump(rex, f, indent=4)
            logging.info(f'[#] <Fit>: Best parameters saved to {PAR_PATH}.')
        logging.info(f'[-] <Fit>: Outcomes fitted for model {model.__name__}.')

    @staticmethod
    def bp_fit(model, matrix: np.ndarray, pred_labels: np.ndarray) -> keras.Model:
        return model(matrix, labels=pred_labels)

    def hc_local_search(self, model, matrix: np.ndarray, pred_labels: np.ndarray, parameters: list, epochs: int = 10):
        __res: (None, np.ndarray) = None
        current_parameters = self.__search(parameters[0], parameters[1:])
        last_parameter = current_parameters[0]
        current_parameter = last_parameter
        repeat_flag = False
        bw = 0.05
        __overall_results = dict()
        for epoch in range(epochs):
            logging.info(f'[#] <Fit> Fitting epoch {epoch + 1}.')
            __local_results = list()
            for current_parameter in current_parameters:
                if current_parameter not in __overall_results:
                    __res = model(matrix, *current_parameter)
                    segmentation = self.__label2seg(__res)
                    wd = window_diff(tf.convert_to_tensor(segmentation), tf.convert_to_tensor(pred_labels))
                    __local_results.append(wd)
                    __overall_results[current_parameter] = float(wd)
            current_parameter = min(__overall_results, key=__overall_results.get)
            current_parameters = self.__search(current_parameter, parameters[1:], bandwidth=bw)
            logging.info(f'[#] <Fit>: Current results:\n{__overall_results}%\nCurrent par.: {current_parameter}.')
            if current_parameter == last_parameter:
                if repeat_flag:
                    break
                bw = bw / 2
                repeat_flag = True
            else:
                repeat_flag = False
            last_parameter = current_parameter
        self.best_parameters = current_parameter

    @staticmethod
    def __label2seg(labels: list[np.ndarray]) -> np.ndarray:
        segs = list()
        for label in labels:
            seg = np.zeros(len(label))
            last_label = -1
            for index, label_number in enumerate(label):
                seg[index] = 1 if label_number != last_label else 0
                last_label = label_number
            segs.append(seg)
        return np.array(segs)

    @staticmethod
    def __search(parameters: tuple, directives: list[tuple], bandwidth: float = 0.05) -> tuple:
        new_parameters = list()
        new_parameters.append(parameters)
        parameter_index = 0
        for directive, parameter in zip(directives, parameters):
            minimum = directive[0]
            maximum = directive[1]
            par_type = directive[2]
            lop = list(parameters)
            if par_type is int:
                parameter_neg = int(parameter - np.ceil(bandwidth * (maximum - minimum)))
                parameter_pos = int(parameter + np.ceil(bandwidth * (maximum - minimum)))
            elif par_type is float:
                parameter_pos = parameter + bandwidth * (maximum - minimum)
                parameter_neg = parameter - bandwidth * (maximum - minimum)
            else:
                lop[parameter_index] = parameter
                new_parameters.append(tuple(lop))
                continue
            if parameter_neg < minimum:
                parameter_neg = minimum
            elif parameter_pos > maximum:
                parameter_pos = maximum
            lop[parameter_index] = parameter_neg
            new_parameters.append(tuple(lop))
            lop[parameter_index] = parameter_pos
            new_parameters.append(tuple(lop))
            parameter_index += 1
        return tuple(new_parameters)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
