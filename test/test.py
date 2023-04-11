# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import logging
from basenet import BaseNetDatabase, window_diff
from src.cosenet import CoSeNet, cosenet_version


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        FUNCTION DEF                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
def main() -> None:
    """
    :return: None
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(f'[+] CoSeNet test connected (v{cosenet_version}).')
    logging.info('[i] Importing database.')
    db = BaseNetDatabase.load('./wikipedia_dataset_256.db')
    db.xtrain = db.xtrain[:][:80][:80]
    db.ytrain = db.ytrain[:][:80]
    db.xtest = db.xtest[:][:80][:80]
    db.ytest = db.ytest[:][:80]
    db.xval = db.xval[:][:80][:80]
    db.yval = db.yval[:][:80]
    logging.info('[i] Creating CoSeNet.')
    cosenet = CoSeNet()
    logging.info('[i] Initial CoSeNet solve test.')
    cosenet.solve(db.xval)
    logging.info('[i] Fitting database.')
    cosenet.fit(5, db.xtrain[:100], db.ytrain[:100])
    logging.info('[i] Testing database.')
    y_hat = cosenet.solve(db.xtest)[1]
    logging.info('[i] Evaluating performance.')
    wd = window_diff(y_hat, db.ytest)
    logging.info(f'[i] WindowDiff: {wd * 100:.2f}%')
    logging.info('[-] CoSeNet disconnected.')


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                           MAIN                            #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
