# AlphaZero gomoku

## How to

### Dependencies
* [Python](https://www.python.org/) 3.5 or higher
* [Pytorch](https://pytorch.org/) build and installed by [installation
script](install.sh)
* [Pybind11](https://pybind11.readthedocs.io/en/stable/) build and installed by [instalation script](install.sh)
* [Cmake](https://cmake.org/) to build python extension
* c++ compiler
* [Gnu Scientific Library](https://www.gnu.org/software/gsl/)

### Instalation

* `git clone --recursive https://github.com/SlavkaMichal/alphazero.git`
* `git clone --recursive git@github.com:SlavkaMichal/alphazero.git`
*  edit `general_config.py` (see [General config](#installation-))
* `./install.sh /path/to/python [/path/to/cuda]`

#### Trouble shooting:
If something went wrong you can install it manually. First you have to `source
env.sh [your python]` to set up eviroment for installation so configuration
files will be found. If you already have Pytorch and Pybind install you have to
set `PREFIX` in [General configuration file](general_config.py) to prefix where
cmake can find them. For more build and installation options look
[here](#installation-).

After finishing configuration you can run `python setup.py install`. File
`setup.py` is located in mcts directory. Additionally you can add installation
prefix `python setup.py install --prefix [PREFIX]`. The installation script is
trying to install the extension to the same prefix where Pytorch and Pybind11
is located.

---
### Monte Carlo tree search extension
Description of implementation details of Monte Carlo tree search extension is
in this [file](mcts/documentation.txt).

---
### Python scripts
Description of implementation details of python scripts is
in this [file](src/documentation.txt).

---
### Configuration

#### General config:
This file contains general configuration for installatin and for running
training process.

##### Self-play configuration:
* `TRAIN_SAMPLES` number of training samples to be genrated during
self-play
* `TIMEOUT_SELF_PLAY` timeuot for self-play

Self-play is terminated when one of these two condition is met.

##### Evaluation configuration:

* `EVAL_GAMES` number of games to be run for evaluation
* `TIMEOUT_EVAL` evaluation timeout
* `EVAL_TRESHOLD` treshold in percatage of games opposing model should win
against current best to be set as best

##### Training configuration:
* `VIEW_STEP` number of times progress is shown per iteration during training
* `WINDOW` tripple determining how many generations of data will be used for
training

##### General:
* `INIT_MOVES` board initialization to a position
* `ROOT` projects root. Don't change this!
* `PARAM_PATH` destination for new parameter parameters
* `DATA_PATH` destination for new data
* `CONFIG_PATH` location where configuration files will be saved for future
reference
* `LOG_PATH` destination for logs
* `PARAM_BEST` file containing information which are currently best parameters
* `LOAD_CONFIG` specifies configuration file that should be loaded (obsolete)
* `LOAD_CONFIG_PARAM` specifies configuration file that is used for best
prarameters in self-play, evaluation, and training
* `LOAD_CONFIG_VERSUS` specifies configuration file that is used for latest
parameters in evaluation
* `THREADS` number of threads which it will run on

* `PROC_NUM` set when multiple processes are run at the same time to avoid
generating files with the same name

##### Installation:
These properties need to be defined before build.

* `PREFIX` prefix where installation script will install binaries
* `INSTALL_PYTORCH` determins if Pytorch should be build and install
* `INSTALL_PYBIND11` determins if Pybind11 should be build and install
* `LOCAL_SITE_PATH` Python modules site
* `PYVERSION` this is for checking Python version
* `MAJOR`, `MINOR` major and minor version
* `SHAPE` dimension of board, current support is only for square boards
* `SIZE` number of positions on the board
* `CUDA` detetermins if Pytorch should be build with cuda support
* `HEUR` building Monte Carlo tree search extension with heuristic
* `DEBUG` enables more debug output


#### Model configuration
In file `config.py` you can confugure model. More deatailed description is in
[configuration.md](configuration.md) file.

---
### Run script:

To run the program I created run script. By passing options you can use almost
all functionality my script  provides. This script is used in combination with
configuration file to have the program behaving the way you want. Some of the
options passed to this script are passed to Python by enviroment variables so
you can do it manually but I wont describe how to do it.

#### Running the script:
* run `./run.sh [ACTION] [OPTIONS]`
#### Actions:
* `-s` or `--self-play`		data generation
* `-t` or `--train`		optimalisation
* `-e` or `--eval`		compare latest parameters with best
* `-g` or `--generation`	creates new file for data
* `-q` or `--set-best`		set latest parameters as the best
* `-r` or `--restore`		restores temporary configuration file
* `-c` or `--config=[FILE]`	sets FILE as root config and stres current configuration in temporary file

#### OPTIONS
* `-p` or `--python`	python interpreter
* `-o` or `--param=[FILE]`	parameter file to be used
* `-v` or `--versus=[FILE]`	oponent parameter file to be used
* `-a` or `--conf-param=[FILE]`	use configuration file
* `-b` or `--conf-versus=[FILE]` use configuration file
* `-n` or `--dry-run`	do not change best parameters
* `-d` or `--data=[FILE|DIR]`	data for training, coma separated list
* `-i` or `--sequence=[NUM]`	sequence nuber, good idea when starting
multiple processes at the same time
* `-h` or `--help`	prints help

#### Examples:
* `./run.sh -s -p /path/to/python -i 0` starts self-play and specifies python
interpreter and appends to the name of log file and generated data file
sequence number 0
* `./run.sh -q -v [FILE with parameters]` sets FILE as best parameters
* `./run.sh -e -p [PARAM1] -a [CONFIG1] -v [PARAM2] -b [CONFIG2] -n` runs
evaluation between parameters PARAM1 and PARAM2 with configuration specified in
CONFIG1 and CONFIG2 files respectively
* `./run.sh -t -d 'DIRECTORY1:DIRECTORY2:path/to/DATA.npy` trains new
parameters using data from DIRECTORY1, DIRECTORY2, and from file DATA.npy

### Using Piskvork tournament manager

1. Start `piskvork.exe` on linux you can use `wine piskvork.exe`.
* In piskvork manager go to `Players->Settings...` and selec player in
`src/pbrain-client-nd.exe`
* Start python server `./pbrain-server.py`
* Play

## Content of the repository

* `src/`              - directory containing python scripts<br>
	* `src/documentation.txt` - documentation for python scripts<br>
* `mcts/`             - sources for MCTS extension<br>
	* `mcts/documentation.txt` - documentation for MCTS extension<br>
	* `mcts/CMakeLists.txt` - documentation for python scripts<br>
	* `mcts/setup.py` - documentation for python scripts<br>
	* `mcts/src/*` - c++ source code for cmcts Python extension<br>
* `general_config.py` - general configuration file <br>
* `config.py`         - configuration for model and tree search<br>
* `install.sh`        - installation script<br>
* `run.sh`            - script for execution of the python scripts<br>
* `env.sh`            - script which sets up enviroment<br>
* `README.md`         - this file<br>
* `CONFIGURATION.md`  - documents configuration options of models and tree search<br>
* `conf/`             - directory for saving configuration files<br>
* `data/`             - directory for generated data<br>
* `logs/`             - directory for log files<br>
* `parameters/`       - directory for optimized parameters<br>
* `.param_best`       - json file containing reference to currently best
parameters<br>

## TODO:
* evaluate also on evaluation dataset
* how to deal with old data:
	1. create separate directory for each generation
		* folder would be named [model name][SHAPE]_[ts]_[generation]
* move getters and setters to lambda functions in module.cpp
* replace run.sh with python script
* how to deal with saved parameters and changes in architecture
	* maybe the best would be to ignore it and if loading fails
	  use initialisation parameters
* move pytorch and pybind to project root
### TODO when nothing else come to mind
* have a look at test, fix them add new
* create evaluation thread
	* node that needs to be evaluated will be locked board and evaluating
will be not blocking right away but will have the node lock until nn is called
* check if cuda works - I am not able to compile pytorch with cuda
* cache (maybe sometime)

