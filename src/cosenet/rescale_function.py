# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np


def rescale_function(input_matrix: np.ndarray, parameter: list[int]):
    wa = parameter[0] * input_matrix
    wb = parameter[1] * 1 / (1 + np.exp(parameter[2] * (25 - 50 * input_matrix)))
    w0 = (1 - parameter[0] - parameter[1]) / 2
    return wa + wb + w0
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
