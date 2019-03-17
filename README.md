# AlphaZero gomoku

## How to:
### Dependencies:
* pybind11:
	1. `git clone https://github.com/pybind/pybind11.git`
	* `source env.sh`
	* `cd pybind11 && pip install --user .`
* libtorch:
	1. `pushd $( python -c 'import config; print(config.PREFIX)')`
	* for cpu <br>
	`wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip` <br>
	 for gpu <br>
	`wget https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-latest.zip`
	<br>
	* `popd`
* cmcts:
	1. `./install.sh`

### Config:
* PREFIX - local installation of python packages

### Generate data:
* adjust config:
	* `SIMS` - MCTS simulation per move
	* `TRAIN_SAMPLES` - aproximate number of training samples to generate
	* `ALPHA` - set alpha for rng or comment it out for default value
	* `CPUCT` - PUCT constant controling exploration in MCTS


## TODO:
* check itf it works with cuda
* multithreaded evaulation
* script for removing duplicated data and adding rotations of board
* cache (maybe sometime)

## Notes:
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
