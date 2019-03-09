## Notes:
* ak to bude padat skusit Custom smart pointre
* skoncil som na 7.8

## TODO:
* navstivit tieto tieto tahy
	* na zaciatku nastavit heuristiku na 0
	* pri vyberani tahov podla heuristiky nebrat v uvahu nevalidne tahy
	* namiesto nulovania heuristickej dosky len odcitat odmenu
* logging
* multiprocessing/ virtual loss
* ulozit metadata o generaciach a iteraciach
* script co nacita ulozene data odstrani duplikaty a prida rotacie

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
