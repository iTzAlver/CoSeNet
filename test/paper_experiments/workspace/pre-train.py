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
import os
import pickle
import json
import numpy as np
import tensorflow as tf
from basenet import BaseNetDatabase, window_diff
from tensorflow import keras
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from transfer import main as transfer_main
MODELS = ['mlp', 'ridge', 'svr']
TPUTS = [8, 16, 32]
VARS = [-1]
PRE_TRAIN_PATH = '../src/cosolver/pre-trained/'
DB_PATH = '../data/sym/'
DB_PATH_STD = '../data/sym_std/'


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def mlp(throughput: int) -> keras.Model:
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(throughput, throughput, 1)))
    divider = 1
    while throughput * throughput / divider > throughput:
        model.add(keras.layers.Dense(throughput * throughput / divider, activation='relu'))
        divider *= 2
    model.add(keras.layers.Dense(throughput, activation='sigmoid'))  # Regression, do not use softmax.
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model


def main():
    for model_name in MODELS:
        for tput in TPUTS:
            results = dict()
            if not os.path.exists(PRE_TRAIN_PATH + f'{model_name}/{tput}/'):
                os.mkdir(PRE_TRAIN_PATH + f'{model_name}/{tput}/')
            for var in VARS:
                if var == -1:
                    alls = [0, 1, 2, 3, 4, 5]
                    dbs = [BaseNetDatabase.load(DB_PATH + f'32k_{tput}t_{_:02d}w.db') for _ in alls]
                    all_x_train = np.concatenate([_.xtrain for _ in dbs])
                    all_x_test = np.concatenate([_.xtest for _ in dbs])
                    all_x_val = np.concatenate([_.xval for _ in dbs])
                    all_x = np.concatenate([all_x_train, all_x_test, all_x_val])
                    all_y_train = np.concatenate([_.ytrain for _ in dbs])
                    all_y_test = np.concatenate([_.ytest for _ in dbs])
                    all_y_val = np.concatenate([_.yval for _ in dbs])
                    all_y = np.concatenate([all_y_train, all_y_test, all_y_val])
                    p = np.random.permutation(len(all_x))
                    all_x = all_x[p]
                    all_y = all_y[p]
                    db = BaseNetDatabase(all_x[:32000], all_y[:32000],
                                         distribution={'train': 70, 'val': 20, 'test': 10}, name='all_var_db')
                    var = 'all_var'
                else:
                    db = BaseNetDatabase.load(DB_PATH + f'32k_{tput}t_{var:02d}w.db')
                logging.info(f'Pre-training {model_name} for {tput}t and {var} variance.')
                if model_name == 'mlp':
                    model = mlp(tput)
                    xtrain = (db.xtrain - np.mean(db.xtrain)) / np.std(db.xtrain)
                    db.xtest = (db.xtest - np.mean(db.xtrain)) / np.std(db.xtrain)
                    model.fit(xtrain, db.ytrain, validation_data=(db.xval, db.yval), epochs=10)
                    model.save(PRE_TRAIN_PATH + f'mlp/{tput}/{var}.h5')
                    mse = float(mean_squared_error(db.ytest, model.predict(db.xtest)))
                    wd = float(window_diff(db.ytest, model.predict(db.xtest)))
                    with open(PRE_TRAIN_PATH + f'mlp/{tput}/std.json', 'w') as f:
                        json.dump({'mean': float(np.mean(db.xtrain)), 'std': float(np.std(db.xtrain))}, f)
                elif model_name == 'ridge':
                    model = Ridge()
                    model.fit(db.xtrain.reshape((-1, tput * tput)), db.ytrain)
                    with open(PRE_TRAIN_PATH + f'ridge/{tput}/{var}.pickle', 'wb') as f:
                        pickle.dump(model, f)
                    mse = float(mean_squared_error(db.ytest, model.predict(db.xtest.reshape((-1, tput * tput)))))
                    wd = float(window_diff(db.ytest, model.predict(db.xtest.reshape((-1, tput * tput)))))
                elif model_name == 'svr':
                    model = Pipeline([('scaler', StandardScaler()), ('svr', MultiOutputRegressor(SVR()))])
                    model.fit(db.xtrain.reshape((-1, tput * tput)), db.ytrain)
                    with open(PRE_TRAIN_PATH + f'svr/{tput}/{var}.pickle', 'wb') as f:
                        pickle.dump(model, f)
                    mse = float(mean_squared_error(db.ytest, model.predict(db.xtest.reshape((-1, tput * tput)))))
                    wd = float(window_diff(db.ytest, model.predict(db.xtest.reshape((-1, tput * tput)))))
                else:
                    raise ValueError(f'Invalid model name {model_name}.')
                results[var] = {'mse': mse, 'wd': wd}
            with open(PRE_TRAIN_PATH + f'{model_name}/{tput}/results.json', 'w') as f:
                json.dump(results, f)
                print(f'Pre-training {model_name} for {tput}t finished: {results}')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    with tf.device('/GPU:0'):
        main()
        transfer_main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
