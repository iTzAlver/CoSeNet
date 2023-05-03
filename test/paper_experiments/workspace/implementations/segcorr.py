# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri as np2ri
from rpy2.robjects import pandas2ri
# You should use the R package SegCorr, which can be installed from CRAN:
#   install.packages("SegCorr")
# You may have trouble installing DNAcopy (a dependency) package, so you can use the following command:
#   if (!require("BiocManager", quietly = TRUE))
#       install.packages("BiocManager")
#   BiocManager::install("DNAcopy")
# And then install SegCorr:
#   install.packages("SegCorr")
# Now you should be able to connect the binding to R and use the package.


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def segcorr(set_of_matrices: np.ndarray, th=0.7) -> np.ndarray:
    logging.info('[+] SegCorr: Importing R binding...')
    corrseg = rpackages.importr('SegCorr')
    rpy2.robjects.r('NA_value <- NA')
    list_of_results = list()
    logging.info('\t[#] SegCorr: Running segments...')
    for corr_matrix in set_of_matrices:
        r_matrix = np2ri.numpy2rpy(corr_matrix)
        _gene_names = np.arange(corr_matrix.shape[-1])
        gene_names = np2ri.numpy2rpy(_gene_names)
        try:
            segmentation = corrseg.segmentation(rpy2.robjects.r['NA_value'], r_matrix, gene_names, th)
            pandas_dataframe = pd.DataFrame(pandas2ri.py2rpy(segmentation[0]))
            segmentations = np.array(pandas_dataframe.iloc[1], dtype=np.int32) - 1
        except Exception as ex:
            logging.warning(ex)
            segmentations = np.array([0])
        list_of_results.append(segmentations)
    logging.info('\t[#] SegCorr: Translating segments to labels (formatting).')
    labels = np.zeros((len(list_of_results), set_of_matrices.shape[-1]))
    for n_d, directive in enumerate(list_of_results):
        for single_directive in directive:
            labels[n_d, single_directive:] += 1
    logging.info('[-] SegCorr: All done.')
    return np.array(labels)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
