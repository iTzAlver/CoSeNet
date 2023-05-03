# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import numpy as np
from sklearn.cluster import KMeans
from tensorflow import keras


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def deep_clustering_unique(set_of_matrix, latent_space: int) -> np.ndarray:
    logging.info("\t[+] Deep Clustering: Formatting data...")
    scaled_corr_matrices = set_of_matrix
    initial_space_len = scaled_corr_matrices.shape[-1]
    scaled_matrices = scaled_corr_matrices.reshape(-1, initial_space_len)
    # Train the autoencoder.
    logging.info("\t[#] Deep Clustering: Building autoencoder...")
    autoencoder, encoder_model, decoder_model = autoencoder_mlp(scaled_matrices, latent_space)
    logging.info("\t[#] Deep Clustering: Fitting autoencoder...")
    autoencoder.fit(scaled_matrices, scaled_matrices, epochs=10, batch_size=32)
    logging.info("\t[#] Deep Clustering: Encoding...")
    encoded_corr_matrices = encoder_model.predict(scaled_matrices)
    decoded_corr_matrices = decoder_model.predict(encoded_corr_matrices).reshape(scaled_corr_matrices.shape)
    decoded_corr_matrices_auto = autoencoder.predict(scaled_matrices).reshape(scaled_corr_matrices.shape)
    assert np.allclose(decoded_corr_matrices, decoded_corr_matrices_auto), "Decoded matrices are not equal."
    logging.info("\t[#] Deep Clustering: Clustering...")
    # Train the clustering model.
    kmeans_labels = KMeans(n_clusters=2, random_state=0, n_init='auto').fit_predict(encoded_corr_matrices)
    logging.info("\t[-] Deep Clustering: All done.")
    labels = np.array(kmeans_labels).reshape(set_of_matrix.shape[0], -1)
    labels[:, 0] = 1
    return labels


def autoencoder_mlp(matrices, latent_space) -> tuple[keras.Model, keras.Model, keras.Model]:
    input_dim = matrices.shape[-1]
    input_layer = keras.layers.Input(shape=(input_dim,))
    encoder = keras.layers.Dense(latent_space, activation='relu')(input_layer)
    decoder = keras.layers.Dense(input_dim, activation='relu')(encoder)
    autoencoder = keras.Model(input_layer, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    encoder_model = keras.Model(input_layer, encoder)
    decoder_model = keras.Model(encoder, decoder)
    return autoencoder, encoder_model, decoder_model


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
