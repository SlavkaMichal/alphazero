import sys
import os

# number of simulations performed by MCTS
SIMS = 3000

# how many training examples should be generated
TRAIN_SAMPLES = 1000

# parameter influencing generating of dirichlet noise
ALPHA =


##############################################################################
#     IF YOU CHANGE ANY OF THE VARIABLES BELLOW YOU MUST RUN ./install.sh    #
##############################################################################

# cmcts install prefix
PREFIX = "{}/bin".format(os.environ["HOME"])

# cmcts site path
CMCTS_SITE_PATH = "{}/lib/python{}.{}/site-packages".format(PREFIX,sys.version_info[0],sys.version_info[1])

# version
MAJOR = 0
MINOR = 1
# dimensions of board
SHAPE = 13

# number of positions on board
SIZE = SHAPE*SHAPE

# compile with support for heuristic and rollout
HEUR = True

# debug output
DEBUG = False

