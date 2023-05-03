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
from basenet import BaseNetDatabase

BASE_DB = r'/home/sugar137/CorNet/db/ht/sym_std/'


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def standardize() -> None:
    """
    :return: None
    """
    logging.basicConfig(level=logging.INFO)
    all_db = [BASE_DB + this for this in os.listdir(BASE_DB)]
    logging.info('[+] Standardizing databases...')
    for db_path in all_db:
        logging.info(f'\tStandardizing: {db_path}')
        db = BaseNetDatabase.load(db_path)
        xtrain = db.xtrain
        xtest = db.xtest
        xval = db.xval
        db.xtrain = (xtrain - xtrain.mean()) / xtrain.std()
        db.xtest = (xtest - xtest.mean()) / xtest.std()
        db.xval = (xval - xval.mean()) / xval.std()
        db.save(db_path)
    logging.info('[-] All done.')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    standardize()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
