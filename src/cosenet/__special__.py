# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import os
__this_path__ = os.path.dirname(os.path.abspath(__file__))
__version__ = '1.3.0'
__pre_train_path__ = os.path.join(__this_path__, 'pre-trained')
__tmp_path__ = os.path.join(__this_path__, 'tmp')

if not os.path.exists(__tmp_path__):
	os.mkdir(__tmp_path__)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
