import sys
import os

##############################################################################
#                                 sefl-play                                  #
##############################################################################
# how many data should be generated
# optimally tens of thousands
TRAIN_SAMPLES = 10000000

# max generating time
TIMEOUT_SELF_PLAY = 1

##############################################################################
#                                 evaluation                                 #
##############################################################################
# number of games run for evaluation
EVAL_GAMES = 50

# max evaluation time in minutes
TIMEOUT_EVAL = 2

# percentual wins for new neural network to replace current best
EVAL_TRESHOLD = 0.6

##############################################################################
#                                    training                                #
##############################################################################
# how many times per iteration will show progress
VIEW_STEP = 20

##############################################################################
#                                   general                                  #
##############################################################################
# project root
ROOT = os.path.dirname(os.path.realpath(__file__))
# path to neural network parameters
PARAM_PATH = "{}/parameters".format(ROOT)

# path to data
DATA_PATH = "{}/data".format(ROOT)

# path to configuration backups
CONFIG_PATH = "{}/conf".format(ROOT)

# path to logs
LOG_PATH = "{}/logs".format(ROOT)

# file containing path to best parameters
PARAM_BEST = "{}/.param_best".format(ROOT)

# number of generations to train from
# size of window is increasing by one every n generations
# (starting window, max window, n)
WINDOW = (4,20,2)

# load config
LOAD_CONFIG = ""
if 'CONFIG' in os.environ:
    LOAD_CONFIG = os.environ['CONFIG']

# special config for parameters
LOAD_CONFIG_PARAM = ""
if 'CONF_PARAM' in os.environ:
    LOAD_CONFIG_PARAM = os.environ['CONF_PARAM']

# special config for oponent
LOAD_CONFIG_VERSUS = ""
if 'CONF_VERSUS' in os.environ:
    LOAD_CONFIG_VERSUS = os.environ['CONF_VERSUS']

THREADS = 4
# number of self-play tasks
PROC_NUM = 0
if 'PROC_NUM' in os.environ:
    PROC_NUM = os.environ['PROC_NUM']

##############################################################################
#     IF YOU CHANGE ANY OF THE VARIABLES BELLOW YOU MUST RUN ./install.sh    #
##############################################################################

# installation prefix
#  meta and merlin
#PREFIX = "{}/.local".format(os.path.dirname(ROOT))
#my pc
PREFIX = "{}/.local".format(os.environ['HOME'])

# test if pytorch is available
INSTALL_PYTORCH = False
#try:
#    import pytorch
#except ModlueNotFoundError:
#    INSTALL_PYTORCH = True

# test if pybind11 is available
INSTALL_PYBIND11 = False
#try:
#    import pybind11
#except ModlueNotFoundError:
#    INSTALL_PYBIND11 = True

# cmcts site path
LOCAL_SITE_PATH = "{}/lib/python{}.{}/site-packages".format(PREFIX,sys.version_info[0],sys.version_info[1])
PYVERSION  = True if sys.version_info[0] >= 3 and sys.version_info[1] > 4 else False

# version
MAJOR = 0
MINOR = 6

# dimensions of board
SHAPE = 13

# number of positions on board
SIZE = SHAPE*SHAPE

# cuda availability
CUDA = False

# compile with support for heuristic and rollout
HEUR = False

# debug output
DEBUG = True
