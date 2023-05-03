# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def herarchical_clustering(set_of_matrices, th, n_clusters) -> np.ndarray:
    """
    :return: None
    """
    logging.info("\t[+] Hierarchical Algorithm: Building instance...")
    _set_of_matrices = np.where(set_of_matrices <= th, 0, set_of_matrices)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labs = list()
    logging.info("\t[#] Hierarchical Algorithm: Computing clusters...")
    for mat in _set_of_matrices:
        laplacian = np.diag(np.sum(mat, axis=1)) - mat
        clustering.fit_predict(laplacian)
        labs.append(clustering.labels_)
    logging.info("\t[-] Hierarchical Algorithm: All done.")
    return np.array(labs)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
