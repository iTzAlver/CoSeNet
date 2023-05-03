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
import time
import tensorflow as tf
from basenet import window_diff
from cosenet import CoSeNet as CorrelationSolver
from implementations import Prepare


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def heuristic() -> None:
    cosenet = CorrelationSolver(throughput=16, model_path='../src/cosolver/pre-trained/ridge/16/5.pickle')
    db = Prepare()
    db_synth = Prepare(synth=True)
    # Fitting the model.
    # logging.info(f'[+] Training the model with the genetic algorithm.')
    # best_genetic = cosenet.fit(20, x=db.matrix[0], y=db.labels[0])[:5]
    # logging.info(f'[+] Training the model with the Particle Swarm algorithm.')
    # best_pso = cosenet.fit(20, x=db.matrix[0], y=db.labels[0], algorithm='pso')[:5]
    best_after_training = [
        # Genetic:
        [0.340457, 0.360398, 0.176421, 0.677903, 16],
        [0.765270, 0.000000, 0.000000, 0.671171, 16],
        [0.638560, 0.000000, 0.000000, 0.599444, 16],
        [0.966205, 0.000000, 0.000000, 0.813063, 16],
        [0.580115, 0.262362, 0.537208, 0.792306, 16],
        # PSO:
        [0.554481, 0.390085, 0.459603, 0.935885, 32],
        [0.555729, 0.389216, 0.461144, 0.935403, 32],
        [0.847700, 0.367580, 0.497995, 0.915899, 32],
        [0.830804, 0.389225, 0.461158, 0.936526, 32],
        [0.822956, 0.371211, 0.464257, 0.909164, 32],
        # CRO:
        [0.1531, 0.1052, 1.0000, 0.6679, 16],
        [1.0000, 0.0000, 1.0000, 1.0000,  8],
        [0.1531, 0.1052, 1.0000, 0.8207, 32]
    ]
    best = 1
    best_parameters = None
    all_results = list()
    for singles in best_after_training[-3:]:
        partial_cosenet = CorrelationSolver(throughput=singles[-1],
                                            model_path=f'../src/cosolver/pre-trained/ridge/{singles[-1]}/5.pickle')
        partial_cosenet.parameters = singles[:-2]
        partial_cosenet.threshold = singles[-2]
        y_hat = partial_cosenet.solve(db.matrix[0])[-1]
        y_ref = db.labels[0]
        wd = window_diff(tf.convert_to_tensor(y_hat), tf.convert_to_tensor(y_ref))
        all_results.append(float(wd))
        if wd < best:
            logging.info(f'[i] New best: {wd}')
            best = wd
            cosenet = partial_cosenet
            best_parameters = singles
    logging.info(f'[i] Best parameters: {best_parameters} with wd: {best}')
    logging.info(f'[i] Average wd: {sum(all_results) / len(all_results)}')
    logging.info(f'[i] All results: {all_results}')

    # Testing the model.
    logging.info(f'[i] Evaluating performance...')
    y_hat_s = [cosenet.solve(mats)[1] for mats in db_synth.matrix[1:]]
    tik = time.perf_counter()
    y_hat = [cosenet.solve(mats)[1] for mats in db.matrix[1:]]
    tok = time.perf_counter()

    # Compute metrics: wd, mse, mae.
    this_results = list()
    for each_matrix_s, each_labels_s in zip(y_hat_s, db_synth.labels[1:]):
        each_matrix_s = tf.convert_to_tensor(each_matrix_s)
        each_labels_s = tf.convert_to_tensor(each_labels_s)
        wd = window_diff(each_matrix_s, each_labels_s)
        mse = tf.keras.losses.MeanSquaredError()(each_matrix_s, each_labels_s)
        mae = tf.keras.losses.MeanAbsoluteError()(each_matrix_s, each_labels_s)
        this_results.append({
            'wd': float(wd),
            'mse': float(mse),
            'mae': float(mae),
            'time': tok - tik,
            'parameters': (*cosenet.parameters, cosenet.threshold, cosenet.throughput),
            'size': each_matrix_s.shape[-1],
            'synth': True
        })
    for each_matrix, each_labels in zip(y_hat, db.labels[1:]):
        each_matrix = tf.convert_to_tensor(each_matrix)
        each_labels = tf.convert_to_tensor(each_labels)
        wd = window_diff(each_matrix, each_labels)
        mse = tf.keras.losses.MeanSquaredError()(each_matrix, each_labels)
        mae = tf.keras.losses.MeanAbsoluteError()(each_matrix, each_labels)
        this_results.append({
            'wd': float(wd),
            'mse': float(mse),
            'mae': float(mae),
            'time': tok - tik,
            'parameters': (*cosenet.parameters, cosenet.threshold, cosenet.throughput),
            'size': int(each_matrix.shape[-1]),
            'synth': False
        })
    with open('../data/our_results.json', 'w') as file:
        json.dump(this_results, file, indent=4)


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    heuristic()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
