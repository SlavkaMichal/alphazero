#!/usr/bin/bash
source env.sh
echo "ARGC $#"

if [ "$#" -ge "1" ]; then
	PYTHON=$1
	CUDA=$($PYTHON -c 'import config; print(config.CUDA)')

	if [ $CUDA == "True" ]; then
		if [ $# != 2 ]; then
			echo "Usage: $0 <python_interpreter> <cuda_root>"
			exit
		fi
		CUDA_PATH=$2
	else
		if [ $# != 1 ]; then
		echo "Usage: $0 <python_interpreter> <cuda_root>"
		fi
	fi
else
	echo "Usage: $0 <python_interpreter> [<cuda_root>]"
	exit
fi
SITE=$($PYTHON -c 'import config; print(config.LOCAL_SITE_PATH)')
PYTORCH=$($PYTHON -c 'import config; print(config.PYTORCH)')
PREFIX=$($PYTHON -c 'import config; print(config.PREFIX)')
DEBUG=$($PYTHON -c 'import config; print(config.DEBUG)')
PYVERSION=$($PYTHON -c 'import config; print(config.PYVERSION)')
#export PYTHONPATH="$PYTHONPATH:$(pwd)"


echo "PYTHON: $PYTHON"
echo "CUDA: $CUDA"
echo "PREFIX: $PREFIX"
if [ $PYVERSION == "False" ]; then
	echo "Python version not high enough need 3.4 or more"
	exit
fi

test config.py -nt mcts/.timestamp && rm -rf mcts/build
test config.py -nt mcts/.timestamp && echo "Removing build directory"

echo "Installing pybind11"
out=$($PYTHON -m pip install --user pybind11)
if [ $DEBUG == "True" ]; then
	echo $out
fi

#pushd $PREFIX > /dev/null
#if [ ! -d libtorch ]; then
#	exit
#	if [ $CUDA == "True" ]; then
#		echo "libtorch with cuda"
#		wget https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-latest.zip
#	else
#		echo "libtorch without cuda"
#		wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
#	fi
#	unzip libtorch-shared-with-deps-latest.zip
#else
#	echo "libtorch installed"
#fi
echo $PYTORCH
if [ $PYTORCH == "True" ]; then
	echo "Installing libtorch"
	pushd pytorch > /dev/null
	git checkout 916a670828bad914907f628e88e6c0ca6bb9b365
	out=$(git submodule update --init --recursive)
	out=$($PYTHON -m pip install --user pyyaml==3.13)
	# typing is not required dependency for python > 3.4
	sed -i '/typing/d' requirements.txt
	out=$($PYTHON -m pip install --user -r requirements.txt)
	if [ $CUDA == "False" ]; then
		USE_OPENCV=1 \
		BUILD_TORCH=ON \
		CMAKE_PREFIX_PATH="/usr/bin/" \
		LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
		USE_CUDA=0 \
		USE_NNPACK=0 \
		CC=cc \
		CXX=c++ \
		$PYTHON setup.py bdist_wheel
		ret=$?
		if [ $ret != 0 ]; then
			exit
		fi
	else
		LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/local/lib:$LD_LIBRARY_PATH \
		CUDA_BIN_PATH=$CUDA_PATH/bin \
		CUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH \
		CUDNN_LIB_DIR=$CUDA_PATH/lib64 \
		CUDA_HOST_COMPILER=cc \
		TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" \
		TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
		USE_OPENCV=1 \
		BUILD_TORCH=ON \
		CMAKE_PREFIX_PATH="/usr/bin/" \
		LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
		USE_CUDA=0 \
		USE_NNPACK=0 \
		CC=cc \
		CXX=c++ \
		$PYTHON setup.py bdist_wheel
		ret=$?
		if [ $ret != 0 ]; then
			exit
		fi
	fi
	$PYTHON -m pip install --update dist/*.whl
	ret=$?
	if [ $ret != 0 ]; then
		echo $out
		exit
	fi

	if [ $DEBUG == "True" ]; then
		echo $out
	fi
	popd > /dev/null
fi

pushd mcts > /dev/null
echo "Installing cmcts module"
out=$($PYTHON setup.py install --prefix $PREFIX)
ret=$?
if [ $ret != 0 ]; then
	echo $out
	exit
fi
if [ $DEBUG == "True" ]; then
	echo $out
fi
popd > /dev/null
touch mcts/.timestamp

if [ $DEBUG == "True" ]; then
	echo "Creating missing directories"
fi
if [ ! -d "model" ]; then
	echo "Creating model directory"
	mkdir "model"
fi
if [ ! -d "data" ]; then
	echo "Creating data directory"
	mkdir "data"
fi
if [ ! -d "logs" ]; then
	echo "Creating logs directory"
	mkdir "logs"
fi
