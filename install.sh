#!/usr/bin/bash
SITE=$(python -c 'import config; print(config.CMCTS_SITE_PATH)')
PREFIX=$(python -c 'import config; print(config.PREFIX)')
#if [[ "$package" ]]
export PYTHONPATH="$PYTHONPATH:$SITE"
echo $PYTHONPATH
test config.py -nt mcts/.timestamp && rm -rf mcts/build
test config.py -nt mcts/.timestamp && echo "Removing build directory"
pushd mcts
python setup.py install --prefix=$PREFIX
popd
touch mcts/.timestamp

echo "Creating missing directories"
if [ !-d "model" ]; then
	echo "Creating model directory"
	mkdir "model"
fi
if [ !-d "data" ]; then
	echo "Creating data directory"
	mkdir "data"
fi
if [ !-d "logs" ]; then
	echo "Creating logs directory"
	mkdir "logs"
fi
