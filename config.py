import sys
import os

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
CONFIG_PATH = "{}/config".format(ROOT)

# path to logs
LOG_PATH = "{}/logs".format(ROOT)

# file containing path to best parameters
PARAM_BEST = "{}/.param_best".format(ROOT)

# load config
LOAD_CONFIG = ""
if 'CONFIG' in os.environ:
    LOAD_CONFIG = os.environ['CONFIG']

# number of self-play tasks
PROC_NUM = 0
if 'PROC_NUM' in os.environ:
    PROC_NUM = os.environ['PROC_NUM']

##############################################################################
#                                neural network                              #
##############################################################################
# neural network to be used
MODEL_MODULE = "model"
MODEL_CLASS = "simplerNN"

# board channels
CHANNELS = 2

# channels in convolutional layers
CONV_CHANNELS = 16

# common convolutional layers
FRONT_LAYER_CNT = 4

# policy head convolutional layer count
POLICY_LAYER_CNT = 2

# value head convolutional layer count
VALUE_LAYER_CNT = 2

##############################################################################
#                                 sefl-play                                  #
##############################################################################
# how many data should be generated
# optimally tens of thousands
TRAIN_SAMPLES = 10000000

# max generating time
TIMEOUT_SELF_PLAY = 15

##############################################################################
#                                 evaluation                                 #
##############################################################################
# number of games run for evaluation
EVAL_GAMES = 50

# max evaluation time in minutes
TIMEOUT_EVAL = 50

# percentual wins for new neural network to replace current best
EVAL_TRESHOLD = 0.6

##############################################################################
#                                    training                                #
##############################################################################
# number of epochs
EPOCHS = 4

# augment data with rotations of board
ROTATIONS = True

# training learning rate
LR = 0.001

# batch size
BATCH_SIZE = 32

# number of generations to train from
# size of window is increasing by one every n generations
# (starting window, max window, n)
WINDOW = (4,20,2)

##############################################################################
#                                    mcts                                    #
##############################################################################
# number of simulations performed by MCTS
SIMS = 1000

# parameter influencing generating of dirichlet noise
# x = avg_game_length = SHAPE*2
# default alpha = 10/((SIZE*x-(x**2+x)*0.5)/x)
# currently using default everywhere
#ALPHA = 0.1

# number of moves after which temperature is set from 1 to infinitesimal value
TAU = 10

# c constant in PUCT algorithm
CPUCT = 4.

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

# number of threads
THREADS = 8

# cuda availability
CUDA = False

# compile with support for heuristic and rollout
HEUR = False

# debug output
DEBUG = True
