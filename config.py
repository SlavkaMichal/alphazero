##############################################################################
#                                neural network                              #
##############################################################################
# neural network to be used
MODEL_MODULE = "model"
MODEL_CLASS = "simplerNN"

# dimensions of board
SHAPE = 13

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

# number of threads
THREADS = 8

