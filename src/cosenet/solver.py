# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import pickle
import logging
import os
import numpy as np
import tensorflow as tf
from basenet import BaseNetGenetic, window_diff, BaseNetDatabase
from basenet.metaheuristic import BaseNetPso
from .rescale_function import rescale_function
from .__special__ import __pre_train_path__, __tmp_path__
# -----------------------------------------------------------


def _callback(epoch_number, population, score):
    logging.info(f'[i] Tracking information:\tEpoch\t\tBest fitness\n\t\t\t\t\t\t\t{epoch_number}\t\t\t{max(score)}')
    return True


def _fitness(*args, **kwargs):
    try:
        solver = CorrelationSolver(throughput=16, model_path=__pre_train_path__ + '/ridge.pickle')
        x = np.load(os.path.join(__tmp_path__, 'x.npy'))
        y = np.load(os.path.join(__tmp_path__, 'y.npy'))
        scores = list()
        for individual in args:
            solver.parameters = [individual[0], individual[1], individual[2]]
            solver.threshold = individual[3]
            solution = solver.solve(x)[1]
            y_hat = tf.convert_to_tensor(solution)
            scores.append(1-solver.fitness_metric(y_hat, tf.convert_to_tensor(y)))
        return tf.convert_to_tensor(scores)
    except Exception as e:
        logging.error(f'[!] Error during the fitness function: {e}')
        raise e


class CorrelationSolver:
    def __init__(self, throughput: int = 16, model_path: str = __pre_train_path__ + '/ridge.pickle'):
        """
        This class is the solver of the correlation matrix segmentation problem.
        :param throughput: Throughput of the algorithm. 16 is the recommended size.
        :param model_path: Path to the model to be used. Ridge model is generally recommended, however, you can use MLP
        or any other model that you can import.
        """
        self.original_size = None
        self.throughput = throughput
        self.rescale = rescale_function
        if '.h5' in model_path:
            self.__model = tf.keras.models.load_model(model_path)
        else:
            with open(model_path, 'rb') as f:
                self.__model = pickle.load(f)
        self.fitness_metric = window_diff
        self.parameters: list[float] = [0.34, 0.36, 0.176]
        self.threshold: float = 0.678

    def solve(self, input_matrix: (np.ndarray, list, tuple)):
        """
        This method solves the problem of the correlation matrix segmentation.
        :param input_matrix: Set of correlation matrices to be segmented.
        :return: A tuple with the correlation matrix without noise and the segmentation boundaries.
        """
        input_matrices = np.array(input_matrix)
        if input_matrices.shape[-1] == 1:
            true_matrices = np.squeeze(input_matrices)
        else:
            true_matrices = input_matrices
        solutions = list()
        segmentation = list()
        for matrix in true_matrices:
            each_solution, each_segmentation = self.__solve(matrix)
            solutions.append(each_solution)
            segmentation.append(each_segmentation)
        return solutions, segmentation

    def __solve(self, true_matrix: np.ndarray):
        self.original_size = true_matrix.shape
        return self.__depth1(true_matrix)

    def fit(self, epochs, x: np.ndarray, y: np.ndarray,
            algorithm='genetic', cores=10, individuals=200, new_individuals=100):
        """
        Fit the model using a metaheuristic. The metaheuristic will try to find the best parameters for the model.
        This method displays a dashboard and makes use of ray distributed computing framework in the machine.
        :param epochs: Number of epochs.
        :param x: Training data.
        :param y: Training labels.
        :param algorithm: Algorithm to be used in the metaheuristic (either genetic or pso)
        :param cores: Number of cores to be used in ray.
        :param individuals: Individuals in the algorithm.
        :param new_individuals: Number of new individuals per epoch.
        :return: The best parameters obtained by the metaheuristic.
        """
        db = BaseNetDatabase(x, y)
        x = np.squeeze(np.concatenate([db.xtrain, db.xtest, db.xval]), axis=-1)
        y = np.concatenate([db.ytrain, db.ytest, db.yval])
        np.save(__tmp_path__ + '/x.npy', x)
        np.save(__tmp_path__ + '/y.npy', y)
        if algorithm == 'genetic':
            bnh = BaseNetGenetic(_fitness,
                                 number_of_individuals=individuals,
                                 new_individuals_per_epoch=new_individuals,
                                 computational_cores=cores,
                                 mutation_variance=0.1)
        else:
            bnh = BaseNetPso(_fitness,
                             number_of_individuals=individuals,
                             new_individuals_per_epoch=new_individuals,
                             computational_cores=cores,
                             inertia=0.1)
        bnh.add_parameter(minimum=0, maximum=1)
        bnh.add_parameter(minimum=0, maximum=1)
        bnh.add_parameter(minimum=0, maximum=1)
        bnh.add_parameter(minimum=0, maximum=1)
        bnh.add_rule([0, 1], [], operator=lambda _x, _: sum(_x) <= 1)
        best = bnh.fit(epochs, callback=_callback)[0][0]
        self.parameters = [float(best[0]), float(best[1]), float(best[2])]
        self.threshold = float(best[3])
        os.remove(__tmp_path__ + '/x.npy')
        os.remove(__tmp_path__ + '/y.npy')
        return best

    def __depth1(self, input_matrix):
        original_size = len(input_matrix)
        r0 = self.__identity_padding__(input_matrix, self.throughput)
        rv = np.array(self.__formatting__(r0, self.throughput))
        s0 = self.__depth2(rv)
        sn = self.__resize_to_original__(s0, original_size)
        output_matrix = self.__reconstruction__(sn)
        return output_matrix, sn

    def __depth2(self, ri: np.ndarray) -> (np.ndarray, None):
        ra = self.__rescale(ri, self.parameters)
        si = self.__depth3(ra)
        sp0 = self.__overlap_mean__(si)
        s0 = self.__threshold__(sp0, self.threshold)
        return s0

    def __depth3(self, ri: np.ndarray) -> np.ndarray:
        mat = ri.reshape(-1, ri.shape[-1] ** 2)
        divergent_prediction = self.__model.predict(mat)
        return divergent_prediction

    # Depth 1:
    @staticmethod
    def __identity_padding__(input_matrix: np.ndarray, throughput: int) -> np.ndarray:
        extension = int((throughput - (len(input_matrix) % throughput)) % throughput) + 1  # Plus one
        # is a performance trick that allows the system to achieve great hit-rate in the latest stages.
        __new_extension = np.zeros((len(input_matrix), extension))
        y_extended = np.append(input_matrix, __new_extension, axis=1)
        _new_extension = np.zeros((extension, len(input_matrix) + extension))
        x_extended = np.append(y_extended, _new_extension, axis=0)
        identity_padded = x_extended.copy()
        for j, _ in enumerate(identity_padded):
            identity_padded[j, j] = 1
        return identity_padded

    @staticmethod
    def __formatting__(input_matrix: np.ndarray, throughput: int) -> tuple:
        slider = round(throughput / 2)
        cursor = 0
        slides = list()

        while cursor + throughput <= input_matrix.shape[0]:
            si = input_matrix[cursor:cursor + throughput, cursor:cursor + throughput]
            slides.append(si)
            cursor += slider

        return tuple(slides)

    @staticmethod
    def __resize_to_original__(input_vector: np.ndarray, size: int) -> np.ndarray:
        return input_vector[0:size]

    @staticmethod
    def __reconstruction__(input_vector: np.ndarray) -> np.ndarray:
        mat = np.zeros((len(input_vector), len(input_vector)), dtype=np.uint8)
        base = 0
        plac_ix = 0
        for idx, element in enumerate(input_vector):
            mat[idx, idx] = 1
            if element == 0:
                for ix in range(base, plac_ix + base + 1):
                    mat[idx, ix] = 1
                    mat[ix, idx] = 1
                plac_ix += 1
            else:
                base = idx
                plac_ix = 0
        return mat

    # Depth 2:
    def __rescale(self, input_matrix: np.ndarray, parameter) -> np.ndarray:
        rescaled = self.rescale(input_matrix, parameter)
        for nc, case in enumerate(rescaled):
            for nr, row in enumerate(case):
                rescaled[nc, nr, nr] = 1.0
        return rescaled

    @staticmethod
    def __overlap_mean__(input_vectors: (list[np.ndarray], np.ndarray)) -> np.ndarray:
        # Remove the one in the beginning.
        formatted_input_vectors = list()
        formatted_input_vectors.append(input_vectors[0])
        middle_point = int(len(input_vectors[0]) / 2)
        for to_be_formatted in input_vectors[1:]:
            to_be_formatted[0] = formatted_input_vectors[-1][middle_point]
            formatted_input_vectors.append(to_be_formatted)
        # Set up the cursor and aliasing.
        middle_cursor = int(input_vectors[0].shape[0] / 2)
        total_length = middle_cursor * (len(input_vectors) + 1)
        placement_array = np.zeros((len(input_vectors), total_length))
        for i, formatted_vector in enumerate(formatted_input_vectors):
            placement_array[i, i * middle_cursor:i * middle_cursor + len(formatted_vector)] = formatted_vector
        placement_array = np.array(placement_array)
        s0 = np.sum(placement_array, axis=0)
        # Divide intersection (mean of 2).
        s0[middle_cursor:-middle_cursor] /= 2
        return s0

    @staticmethod
    def __threshold__(input_vector: np.ndarray, threshold: float):
        iv = input_vector.copy()
        iv[iv < threshold] = 0
        iv[iv >= threshold] = 1
        return iv

    # Override methods:
    def __repr__(self):
        __text__ = f'<CorrelationSolver instance with {self.throughput} throughput>'
        return __text__

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
