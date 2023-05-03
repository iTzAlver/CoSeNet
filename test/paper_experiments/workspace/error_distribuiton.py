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
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from basenet import BaseNetDatabase
from cosenet import CoSeNet as CorrelationSolver


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main() -> None:
    with open('../src/cosolver/pre-trained/ridge/8/5.pickle', 'rb') as _file:
        inner_model = pickle.load(_file)
    overall_model = CorrelationSolver(throughput=8, model_path='../src/cosolver/pre-trained/ridge/8/5.pickle')
    overall_model.parameters = [1, 0, 0]
    overall_model.threshold = 0.5

    db_32 = BaseNetDatabase.load('../data/sym/32k_32t_02w.db')
    samples = db_32.xtest
    solutions = db_32.ytest
    predictions = overall_model.solve([samples])[1]
    np_errors = np.mean(1 - np.abs(predictions - solutions), axis=0)

    split_samples = list()
    split_solutions = list()
    split_predictions = list()
    for sample, solution in zip(samples, solutions):
        m1 = np.ceil(sample.shape[0] / 4) - 1
        _split_samples = list()
        _split_solutions = list()
        _split_predictions = list()
        for _ in range(int(m1)):
            current_sample = np.squeeze(sample[_ * 4:_ * 4 + 8, _ * 4:_ * 4 + 8])
            _split_samples.append(current_sample)
            current_sol = np.copy(solution[_ * 4:_ * 4 + 8])
            current_sol[0] = 1
            _split_solutions.append(current_sol)
            in_s = current_sample.reshape(1, -1)
            _split_predictions.append(inner_model.predict(in_s)[0])
        split_samples.append(_split_samples)
        split_solutions.append(_split_solutions)
        split_predictions.append(_split_predictions)

    np_split_predictions = np.array(split_predictions)
    np_split_predictions = np.where(np_split_predictions > 0.5, 1, 0)
    np_split_solutions = np.array(split_solutions)
    np_split_errors = np.mean(1 - np.abs(np_split_predictions - np_split_solutions), axis=0)

    overall_error = np.zeros(32)
    all_errors = list()
    for pre_i, error_vec in enumerate(np_split_errors):
        _error_vec = error_vec.copy()
        preamble = np.zeros(pre_i * 4)
        posamble = np.zeros((6 - pre_i) * 4)
        _error_vec[0] = 0
        error = np.concatenate((preamble, _error_vec, posamble))
        overall_error += error
        all_errors.append(error)
    overall_error /= 2

    plt.figure(figsize=(10, 5))
    np_errors[0] = 0
    sns.lineplot(np_errors, color='r', linewidth=2, alpha=0.5, label='Overall success')
    for _, each_error in enumerate(all_errors):
        plt.axvline(x=[4 * (_ + 1)], color='k', linestyle='--')
        plt.fill_between(np.linspace(0, 31, 32), each_error, color='c', alpha=0.1)
        sns.lineplot(x=np.linspace(0, 31, 32), y=each_error, color='b', linewidth=0.5, linestyle='--')

    plt.ylim(0.6, 1)
    plt.xlabel('Segmentation prediction index')
    plt.ylabel('1 - MAE')
    plt.title('Success distribution.')
    plt.xticks(np.linspace(0, 31, 32))
    plt.legend(['Overall success', '_nolegend_', 'Sub-matrix center', 'Sub-matrix success'], loc='lower right')
    plt.grid()
    plt.xlim(1, 31)
    plt.savefig('../render/error_distribution.png')
    plt.show()

    print('[-] Disconnected...')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
