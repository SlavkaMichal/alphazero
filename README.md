# AlphaZero gomoku

## How to:
### Instalation:
* `git clone --recursive https://github.com/SlavkaMichal/alphazero.git`
* `git clone --recursive git@github.com:SlavkaMichal/alphazero.git`
* `./install.sh /python/path`
* Note: at this moment `cmake_prefix_path` to pybind11 and pytorch is not
deduced automatically you have to adjust it in `mcts/CMakeLists.txt`
accordingly

### Config:
* PREFIX - local installation of python packages and other dependecies

### Generate data:
* adjust config:
	* `SIMS` - MCTS simulation per move
	* `TRAIN_SAMPLES` - aproximate number of training samples to generate
	* `ALPHA` - set alpha for rng or comment it out for default value
	* `CPUCT` - PUCT constant controling exploration in MCTS
* run `src/self-play.py`


## TODO:
* create evaluation thread
* check if cuda works
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
