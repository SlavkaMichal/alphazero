import sys
import os

# number of simulations performed by MCTS
SIMS = 3000

# how many training examples should be generated
# optimally tens of thousands
TRAIN_SAMPLES = 1

# parameter influencing generating of dirichlet noise
# x = avg_game_length = SHAPE*2
# alpha = 10/((SIZE*x-(x**2+x)*0.5)/x)
ALPHA = 0.1

# c constant in PUCT algorithm
CPUCT = 4.

# number of epochs
EPOCHS = 2

# number of generations to train from
# size of window is increasing by one every n generations
# (starting window, max window, n)
WINDOW = (4,2,20)

##############################################################################
#     IF YOU CHANGE ANY OF THE VARIABLES BELLOW YOU MUST RUN ./install.sh    #
##############################################################################

# cmcts install prefix
PREFIX = "{}/.local/bin".format(os.environ["HOME"])

# cmcts site path
LOCAL_SITE_PATH = "{}/lib/python{}.{}/site-packages".format(PREFIX,sys.version_info[0],sys.version_info[1])

# version
MAJOR = 0
MINOR = 2
# dimensions of board
SHAPE = 13

# number of positions on board
SIZE = SHAPE*SHAPE

# compile with support for heuristic and rollout
HEUR = True

# debug output
DEBUG = False

