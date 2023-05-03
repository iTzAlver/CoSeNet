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
import json
import tensorflow as tf
from tensorflow import keras
from basenet import BaseNetDatabase, window_diff, BaseNetLMSE
from tensorflow_addons.metrics import RSquare
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
BASE_DB = r'../data/sym_std/'
ALL_DB_STD = [BASE_DB + this for this in os.listdir(BASE_DB) if '64' not in this]
BASE_DB = r'../data/sym/'
ALL_DB = [BASE_DB + this for this in os.listdir(BASE_DB) if '64' not in this]
logging.basicConfig(level=logging.INFO)
RESULTS_PATH = '../data/regressors.json'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def mlp(throughput: int) -> keras.Model:
    """
    :return: None
    """
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(throughput, throughput, 1)))
    divider = 1
    while throughput * throughput / divider > throughput:
        model.add(keras.layers.Dense(throughput * throughput / divider, activation='relu'))
        divider *= 2
    model.add(keras.layers.Dense(throughput, activation='sigmoid'))  # Regression, do not use softmax.
    model.compile(optimizer='adam', loss='mse', metrics=[RSquare(), 'mse', 'mae'])
    return model


def cnn(throughput: int) -> keras.Model:
    """
    :return: None
    """
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(throughput, (3, 3), activation='relu', input_shape=(throughput, throughput, 1)))
    model.add(keras.layers.Conv2D(throughput / 2, (3, 3), activation='relu', strides=2))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(throughput * 4, activation='relu'))
    model.add(keras.layers.Dense(throughput, activation='sigmoid'))  # Regression, do not use softmax.
    model.compile(optimizer='adam', loss='mse', metrics=[RSquare(), 'mse', 'mae'])
    return model


def main():
    with open(RESULTS_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    logging.info('[+] Loaded current data.')
    for db_path, db_path_std in zip(ALL_DB, ALL_DB_STD):
        db = BaseNetDatabase.load(db_path)
        db_std = BaseNetDatabase.load(db_path_std)
        throughput = int(db.xtrain.shape[1])
        variance = int(db_path.split('_')[-1].split('.')[0].replace('w', ''))
        logging.info(f'\t - Processing {throughput}x{throughput} with {variance} variance...')
        # Non-standardized:
        cnn_model = cnn(throughput)
        mlp_model = mlp(throughput)
        svd_model = BaseNetLMSE(db)
        cnn_model.fit(db.xtrain, db.ytrain, validation_data=(db.xval, db.yval), epochs=10)
        mlp_model.fit(db.xtrain, db.ytrain, validation_data=(db.xval, db.yval), epochs=10)
        y_hat = cnn_model.predict(db.xtest)
        data[f'{throughput}_{variance}']['cnn'] = {'mse': float(mean_squared_error(db.ytest, y_hat)),
                                                   'mae': float(mean_absolute_error(db.ytest, y_hat)),
                                                   'wd': float(window_diff(db.ytest, y_hat)),
                                                   'r2': float(r2_score(db.ytest, svd_model.predict(db.xtest)))}
        y_hat = mlp_model.predict(db.xtest)
        data[f'{throughput}_{variance}']['mlp'] = {'mse': float(mean_squared_error(db.ytest, y_hat)),
                                                   'mae': float(mean_absolute_error(db.ytest, y_hat)),
                                                   'wd': float(window_diff(db.ytest, y_hat)),
                                                   'r2': float(r2_score(db.ytest, svd_model.predict(db.xtest)))}
        y_hat = svd_model.predict(db.xtest)
        data[f'{throughput}_{variance}']['svd'] = {'mse': float(mean_squared_error(db.ytest, y_hat)),
                                                   'mae': float(mean_absolute_error(db.ytest, y_hat)),
                                                   'wd': float(window_diff(db.ytest, y_hat)),
                                                   'r2': float(r2_score(db.ytest, svd_model.predict(db.xtest)))}
        # Standardized:
        cnn_model = cnn(throughput)
        mlp_model = mlp(throughput)
        svd_model = BaseNetLMSE(db_std)
        cnn_model.fit(db_std.xtrain, db_std.ytrain, validation_data=(db_std.xval, db_std.yval), epochs=10)
        mlp_model.fit(db_std.xtrain, db_std.ytrain, validation_data=(db_std.xval, db_std.yval), epochs=10)
        y_hat = cnn_model.predict(db_std.xtest)
        data[f'{throughput}_{variance}']['cnn_scaled'] = \
            {'mse': float(mean_squared_error(db_std.ytest, y_hat)),
             'mae': float(mean_absolute_error(db_std.ytest, y_hat)),
             'wd': float(window_diff(db_std.ytest, y_hat)),
             'r2': float(r2_score(db_std.ytest, svd_model.predict(db_std.xtest)))}
        y_hat = mlp_model.predict(db_std.xtest)
        data[f'{throughput}_{variance}']['mlp_scaled'] = \
            {'mse': float(mean_squared_error(db_std.ytest, y_hat)),
             'mae': float(mean_absolute_error(db_std.ytest, y_hat)),
             'wd': float(window_diff(db_std.ytest, y_hat)),
             'r2': float(r2_score(db_std.ytest, svd_model.predict(db_std.xtest)))}
        y_hat = svd_model.predict(db_std.xtest)
        data[f'{throughput}_{variance}']['svd_scaled'] = \
            {'mse': float(mean_squared_error(db_std.ytest, y_hat)),
             'mae': float(mean_absolute_error(db_std.ytest, y_hat)),
             'wd': float(window_diff(db_std.ytest, y_hat)),
             'r2': float(r2_score(db_std.ytest, svd_model.predict(db_std.xtest)))}
        with open(RESULTS_PATH, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
    logging.info('[-] Saved results.')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    with tf.device('/GPU:0'):
        main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
