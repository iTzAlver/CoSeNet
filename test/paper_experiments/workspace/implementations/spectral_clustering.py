# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import numpy as np
from sklearn.cluster import SpectralClustering


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def correlation_clustering(set_of_matrices, th, n_clusters) -> np.ndarray:
    # Calculate the Laplacian of the correlation matrix
    logging.info("\t[+] SpectralClustering Algorithm: Building instance...")
    _set_of_matrices = np.where(set_of_matrices <= th, 0, set_of_matrices)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    labs = list()
    logging.info("\t[#] SpectralClustering Algorithm: Computing clusters...")
    for mat in _set_of_matrices:
        clustering.fit(mat)
        labs.append(clustering.labels_)
    logging.info("\t[-] SpectralClustering Algorithm: All done.")
    return np.array(labs)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
