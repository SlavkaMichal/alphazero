# Configuration of model and tree search

Configuration of model and tree search is specified in file `config.py`. You
can use run script to specify different configuration file or change
by changing variable in the prefiously mentioned file.

## Variables in configuration file:

### Architecture of neural network

* `MODEL_MODULE` module containing definition of neural the neural network
* `MODEL_CLASS`  class name of neural network from module
* `SHAPE`        one dimension of a input of neural network
* `CHANNELS`     depth of the input
* `CONV_CHANNELS` number of channels in each convolutional layer
* `FRONT_LAYER_CNT` number of layers in common part of neural network
* `POLICY_LAYER_CNT` number of layers in policy head
* `VALUE_LAYER_CNT` number of layers in value head

### Training
* `EPOCHS` number of epochs on training data during trainig
* `ROTATIONS` train on rotations of boad positions
* `LR` learning rate
* `BATCH_SIZE` batch size

### Monte Carlo tree search

* `SIMS` number of simulation performed for each move selection
* `TIMEOUT` timeuot for simulations in seconds
* `EPS` coeficient setting significance of Dirichlet noise
* `TAU` number of moves after which temperature will be set to infinitesimal
number
* `CPUCT` constant influencing level of exploration
* `USE_NN` if set to True neural network is used, else MCTS uses heuristics
