# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import numpy as np


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def proximity_based_merging_means(set_of_matrix, th, oim, cbt) -> np.ndarray:
    logging.info("\t[+] Proximity Based Merging Means algorithm: Obtaining directives...")
    directives = [pbmm(single_mat, (th, oim, cbt)) for single_mat in set_of_matrix]
    logging.info("\t[#] Proximity Based Merging Means algorithm: Translating directives to labels...")
    labels = np.zeros((len(directives), set_of_matrix.shape[-1]))
    for n_d, directive in enumerate(directives):
        for single_directive in directive:
            labels[n_d, single_directive:] += 1
    logging.info("\t[-] Proximity Based Merging Means algorithm: All done.")
    return labels


def pbmm(r: np.array, param: tuple):
    """
    PBMM algorithm.
    :param r: Correlation matrix.
    :param param: Parameters of the algorithm: (th, OIM, cbt)
    :return: List with the segmentation directives.
    """
    # Parameter check:
    if len(param) == 3:
        th = param[0]
        oim = param[1]
        cbt = param[2]
    else:
        raise Exception(f'Invalid parameters in PBMM algorithm: {len(param)} given 3 expected.')

    # Variable initialization:
    failure_counter = 0
    last_index = -1
    current_index = 0
    appindex = 0
    d = []

    # Algorithm loop:
    while (current_index := current_index + 1) < len(r):
        # Compute mean
        elements = r[current_index][last_index+1:current_index]
        if current_index - last_index - 1 <= 0:
            mean = 1
        else:
            mean = sum(elements) / (current_index - last_index - 1)
        # Algorithm control:
        if mean < th:
            failure_counter += 1
        else:
            appindex = current_index
            failure_counter = 0

        if failure_counter > oim:
            d.append(appindex + 1)
            len_cb = d[-1] - last_index - 1     # Checkback init.
            init_cb = last_index + 1            # Checkback init.
            last_index = appindex
            current_index = appindex

            # Checkback...
            if len_cb > 1:
                cb_mean = 0
                for i in range(len_cb - 1):
                    cb_mean += r[init_cb][init_cb + i + 1]
                cb_mean /= (len_cb - 1)
                if cb_mean < cbt:
                    # Check back integrity...
                    cb_mean_back = r[init_cb][init_cb - 1] if init_cb > 0 else -1
                    if cb_mean_back < cbt:
                        aux = d.pop(-1)
                        if d:
                            d.append(d[-1] + 1)
                        else:
                            d.append(1)
                        d.append(aux)
                    else:
                        if d:
                            d[-2] += 1
                        else:
                            d.append(1)
    # Last element:
    d.append(current_index)
    return d
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
