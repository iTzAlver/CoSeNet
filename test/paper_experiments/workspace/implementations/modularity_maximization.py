# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def modularity_maximization(set_of_matrices, threshold=0.25) -> np.ndarray:
    """
    :return: None
    """
    # convert correlation matrix to a graph
    set_of_matrices = np.where(set_of_matrices <= threshold, 0, set_of_matrices)
    set_of_matrices = np.where(set_of_matrices > threshold, set_of_matrices ** -1, set_of_matrices)
    logging.info("\t[+] Modularity maximization algorithm: Obtaining graphs...")
    graphs = [nx.from_numpy_array(matrix) for matrix in set_of_matrices]
    # apply greedy modularity maximization
    logging.info("\t[#] Modularity maximization algorithm: Obtaining communities...")
    communities = [list(greedy_modularity_communities(graph)) for graph in graphs]
    # assign cluster labels based on the communities
    logging.info("\t[#] Modularity maximization algorithm: Assigning cluster labels...")
    list_cluster_labels = list()
    for community in communities:
        cluster_labels = np.zeros(set_of_matrices.shape[-1], dtype=int)
        for i, comm in enumerate(community):
            for node in comm:
                cluster_labels[node] = i
        list_cluster_labels.append(cluster_labels)
    logging.info("\t[-] Modularity maximization algorithm: All done.")
    return np.array(list_cluster_labels)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
