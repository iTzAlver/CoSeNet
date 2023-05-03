# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import numpy as np
from tensorflow import keras


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def deep_clustering(set_of_matrix, labels=None, type_enc='cnn') -> np.ndarray:
    logging.info("\t[+] Deep Clustering: Formatting data...")
    scaled_corr_matrices = set_of_matrix
    # Train the autoencoder.
    logging.info("\t[#] Deep Clustering: Building autoencoder...")
    if type_enc == 'cnn':
        autoencoder, encoder_model, decoder_model = autoencoder_cnn(scaled_corr_matrices, 16)
        scaled_matrices = scaled_corr_matrices
    else:
        autoencoder, encoder_model, decoder_model = autoencoder_mlp(scaled_corr_matrices, 16)
        scaled_matrices = scaled_corr_matrices.reshape(-1, scaled_corr_matrices.shape[-1])
    logging.info("\t[#] Deep Clustering: Fitting autoencoder...")
    if type_enc == 'cnn':
        n_of_mats = scaled_corr_matrices.shape[0]
        sharding = int(np.ceil(n_of_mats / 20))
        for i in range(20):
            ini = i * sharding
            end = (i + 1) * sharding
            if i < 19:
                autoencoder.fit(scaled_matrices[ini:end], scaled_matrices[ini:end], epochs=25, batch_size=32)
            else:
                autoencoder.fit(scaled_matrices[ini:], scaled_matrices[ini:], epochs=25, batch_size=32)
    else:
        autoencoder.fit(scaled_matrices, scaled_matrices, epochs=100, batch_size=32)

    logging.info("\t[#] Deep Clustering: Encoding...")
    encoded_corr_matrices = encoder_model.predict(scaled_matrices)
    decoded_corr_matrices = decoder_model.predict(encoded_corr_matrices).reshape(scaled_corr_matrices.shape)
    decoded_corr_matrices_auto = autoencoder.predict(scaled_matrices).reshape(scaled_corr_matrices.shape)
    assert np.allclose(decoded_corr_matrices, decoded_corr_matrices_auto)
    if type_enc == 'cnn':
        encoded_corr_matrices = np.concatenate(encoded_corr_matrices, axis=0).reshape(-1, 16)

    logging.info("\t[#] Deep Clustering: Training classifier...")
    # Train the clustering model.
    if labels is not None:
        clustering_model = classifier(encoded_corr_matrices.shape[-1])
        clustering_model.fit(encoded_corr_matrices, labels.reshape(-1), epochs=20, batch_size=32)
        clustering_model.save(f'./temp/classifier_{type_enc}.h5')
    else:
        clustering_model = keras.models.load_model('./temp/classifier.h5')

    logging.info("\t[#] Deep Clustering: Predicting labels...")
    end_labels = clustering_model.predict(encoded_corr_matrices)
    end_labels = np.where(end_labels > np.mean(end_labels) * 1.37, 1, 0)
    end_labels = end_labels.reshape(set_of_matrix.shape[0], set_of_matrix.shape[-1])
    return end_labels


def autoencoder_cnn(scaled_corr_matrices, latent_space):
    input_dim = scaled_corr_matrices.shape[-1]
    input_layer = keras.layers.Input(shape=(input_dim, input_dim, 1))
    reshape_layer = keras.layers.Reshape((input_dim, input_dim, 1))(input_layer)
    conv_layer = keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(reshape_layer)
    max_pool = keras.layers.MaxPooling2D((3, 3))(conv_layer)
    max_pool = keras.layers.MaxPooling2D((3, 3))(max_pool)
    flatten_layer = keras.layers.Flatten()(max_pool)
    dense_0 = keras.layers.Dense(input_dim * latent_space, activation='relu')(flatten_layer)
    encoder = keras.layers.Reshape((input_dim, latent_space, 1))(dense_0)
    flatten2_layer = keras.layers.Flatten()(encoder)
    dense_2 = keras.layers.Dense(input_dim * input_dim, activation='relu')(flatten2_layer)
    decoder = keras.layers.Reshape((input_dim, input_dim, 1))(dense_2)
    autoencoder = keras.Model(input_layer, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder_model = keras.Model(input_layer, encoder)
    decoder_model = keras.Model(encoder, decoder)
    return autoencoder, encoder_model, decoder_model


def autoencoder_mlp(scaled_corr_matrices, latent_space):
    input_dim = scaled_corr_matrices.shape[-1]
    encoding_dim = latent_space
    input_layer = keras.layers.Input(shape=(input_dim,))
    reshape_layer = keras.layers.Flatten()(input_layer)
    encoder = keras.layers.Dense(encoding_dim, activation='relu')(reshape_layer)
    decoder = keras.layers.Dense(input_dim, activation='sigmoid')(encoder)
    autoencoder = keras.Model(input_layer, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder_model = keras.Model(input_layer, encoder)
    decoder_model = keras.Model(encoder, decoder)
    return autoencoder, encoder_model, decoder_model


def classifier(size):
    input_layer = keras.layers.Input(shape=(size,))
    dense_layer = keras.layers.Dense(size / 2, activation='relu')(input_layer)
    dense_layer = keras.layers.Dense(16, activation='relu')(dense_layer)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(dense_layer)
    classifier_model = keras.Model(input_layer, output_layer)
    classifier_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier_model
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
