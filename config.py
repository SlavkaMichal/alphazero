import sys
import os

##############################################################################
#                                   general                                  #
##############################################################################
# neural network to be used
MODEL_MODULE = "model"
MODEL_CLASS = "simplerNN"

# path to neural network parameters
PARAM_PATH = "{}/model".format(os.path.dirname(os.path.realpath(__file__)))

# path to data
DATA_PATH = "{}/data".format(os.path.dirname(os.path.realpath(__file__)))

# file containing path to best parameters
PARAM_BEST = "{}/.param_best".format(os.path.dirname(os.path.realpath(__file__)))

##############################################################################
#                                     mcts                                   #
##############################################################################
# number of simulations performed by MCTS
SIMS = 800

# how many training examples should be generated
# optimally tens of thousands
TRAIN_SAMPLES = 8000

# parameter influencing generating of dirichlet noise
# x = avg_game_length = SHAPE*2
# alpha = 10/((SIZE*x-(x**2+x)*0.5)/x)
ALPHA = 0.1

# c constant in PUCT algorithm
CPUCT = 4.

##############################################################################
#                                    training                                #
##############################################################################

# number of epochs
EPOCHS = 2

# training learning rate
LR = 0.001

# batch size
BATCH_SIZE = 16

# number of generations to train from
# size of window is increasing by one every n generations
# (starting window, max window, n)
WINDOW = (4,20,2)

##############################################################################
#     IF YOU CHANGE ANY OF THE VARIABLES BELLOW YOU MUST RUN ./install.sh    #
##############################################################################

# installation prefix
PREFIX = "{}/.local".format(os.environ["HOME"])

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
MINOR = 4

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
DEBUG = False

# compile python extension
EXTENSION = False
