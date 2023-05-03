# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import pickle
import json
import time
import os
from basenet import BaseNetDatabase, window_diff
from tensorflow import keras
from sklearn.metrics import mean_squared_error

PRE_TRAIN_PATH = '../src/cosolver/pre-trained/'
DB_PATH = '../data/sym/'
DB_PATH_STD = '../data/sym_std/'
SAVE_PATH = '../data/transfer.json'

MODELS = ['mlp', 'ridge', 'svr']
TPUTS = [8, 16, 32]
_VARS = [-1]
VARS = [0, 1, 2, 3, 4, 5]


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main() -> None:
    """
    :return: None
    """
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(SAVE_PATH):
        results = list()
    else:
        with open(SAVE_PATH, 'r') as f:
            results = json.load(f)
    for model_name in MODELS:
        for tput in TPUTS:
            for inner_variance in _VARS:
                if inner_variance == -1:
                    inner_variance_name = 'all_var'
                else:
                    inner_variance_name = str(inner_variance)
                tik = time.perf_counter()
                wd, mse, var = None, None, None
                if model_name == 'mlp':
                    model = keras.models.load_model(PRE_TRAIN_PATH + f'{model_name}/{tput}/{inner_variance_name}.h5')
                    with open(PRE_TRAIN_PATH + f'{model_name}/{tput}/std.json', 'rb') as f:
                        std_data = json.load(f)
                        std_mean = std_data['mean']
                        std_std = std_data['std']
                    for var in VARS:
                        this_db = BaseNetDatabase.load(DB_PATH + f'32k_{tput}t_{var:02d}w.db')
                        xtest = (this_db.xtest - std_mean) / std_std
                        y_hat = model.predict(xtest)
                        wd = window_diff(y_hat, this_db.ytest)
                        mse = mean_squared_error(y_hat, this_db.ytest)
                elif model_name == 'ridge':
                    with open(PRE_TRAIN_PATH + f'{model_name}/{tput}/{inner_variance_name}.pickle', 'rb') as f:
                        model = pickle.load(f)
                    for var in VARS:
                        this_db = BaseNetDatabase.load(DB_PATH + f'32k_{tput}t_{var:02d}w.db')
                        y_hat = model.predict(this_db.xtest.reshape((-1, tput * tput)))
                        wd = window_diff(y_hat, this_db.ytest)
                        mse = mean_squared_error(y_hat, this_db.ytest)
                elif model_name == 'svr':
                    with open(PRE_TRAIN_PATH + f'{model_name}/{tput}/{inner_variance_name}.pickle', 'rb') as f:
                        model = pickle.load(f)
                    for var in VARS:
                        this_db = BaseNetDatabase.load(DB_PATH + f'32k_{tput}t_{var:02d}w.db')
                        y_hat = model.predict(this_db.xtest.reshape((-1, tput * tput)))
                        wd = window_diff(y_hat, this_db.ytest)
                        mse = mean_squared_error(y_hat, this_db.ytest)
                else:
                    raise ValueError('Unknown model')
                results.append({'model': str(model_name), 'throughput': int(tput),
                                'inner': float(inner_variance), 'target': float(var),
                                'wd': float(wd), 'mse': float(mse)})
                tok = time.perf_counter()
                print('Tik-tok: ', tok - tik)
                with open(SAVE_PATH, 'w') as f:
                    json.dump(results, f)
                    logging.info(f'Finished {model_name} {tput}t {inner_variance}w')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
