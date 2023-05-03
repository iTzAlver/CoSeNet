# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import community
import networkx as nx
import numpy as np


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def community_detection(set_of_matrices, threshold=0.25) -> np.ndarray:
    set_of_matrices = np.where(set_of_matrices <= threshold, 0, set_of_matrices)
    set_of_matrices = np.where(set_of_matrices > threshold, set_of_matrices ** -1, set_of_matrices)
    logging.info("\t[+] Community detection algorithm: Obtaining graphs...")
    graphs: list = [nx.from_numpy_array(test) for test in set_of_matrices]
    logging.info("\t[#] Community detection algorithm: Obtaining communities...")
    partitions: list = [community.best_partition(graph) for graph in graphs]
    logging.info("\t[#] Community detection algorithm: Assigning cluster labels...")
    res = np.array([partition.values() for partition in partitions])
    logging.info("\t[-] Community detection algorithm: All done.")
    return res
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
