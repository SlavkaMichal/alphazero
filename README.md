# AlphaZero gomoku

## How to:
### Instalation:
* `git clone --recursive https://github.com/SlavkaMichal/alphazero.git`
* `git clone --recursive git@github.com:SlavkaMichal/alphazero.git`
* `./install.sh /python/path`
* Note: at this moment `cmake_prefix_path` to pybind11 and pytorch is not
deduced automatically you have to adjust it in `mcts/CMakeLists.txt`
accordingly

#### Config:
* PREFIX - local installation of python packages and other dependecies

### Run script:
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


### Generate data:
* adjust config:
	* `SIMS` - MCTS simulation per move
	* `TRAIN_SAMPLES` - aproximate number of training samples to generate
	* `ALPHA` - set alpha for rng or comment it out for default value
	* `CPUCT` - PUCT constant controling exploration in MCTS
* run `src/self-play.py`


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

## Notes:
* `python: ../nptl/pthread_mutex_lock.c:352: __pthread_mutex_lock_full: Assertion INTERNAL_SYSCALL_ERRNO (e, __err) != ESRCH || !robust' failed.`
	* this means that the thread that takes lock dies and does not release
it, but I use scoped locks so where's the catch?
* `94635 Illegal instruction`
	* probably cpu does not support binary
* `terminate called after throwing an instance of 'std::system_error'`
	* seems this is occures during calls to pytorch lib
* `python: ../nptl/pthread_mutex_lock.c:117: __pthread_mutex_lock: Assertion mutex->__data.__owner == 0' failed.`



* if there are segfaults between cpp/python interface try smart pointers

## TRAINING PIPELINE:
* Self-play:
	* Nacita najlepsi model alebo ho ulozi
	* `../model/[model name][SHAPE]_[iteracia].pt` alebo `../model/best`
	* data ulozi do `../data/[model name][SHAPE]_[idx]_[iteracia]`

* Train:
	* trenuje najlepsi model na najnovsich datach
	* nacita data `../data/[model name][SHAPE]_[idx]_[iteracia]`
	* idx sa bude linearne zvacsovat podla configu
	* iteracia je irelevantna
