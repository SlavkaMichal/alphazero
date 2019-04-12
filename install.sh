#!/bin/bash

if [ "$#" -ge "1" ]; then
	PYTHON=$1
	source env.sh $PYTHON
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
INSTALL_PYTORCH=$($PYTHON -c 'import config; print(config.INSTALL_PYTORCH)')
PARAM_BEST=$($PYTHON -c 'import config; print(config.PARAM_BEST)')
INSTALL_PYBIND11=$($PYTHON -c 'import config; print(config.INSTALL_PYBIND11)')
PREFIX=$($PYTHON -c 'import config; print(config.PREFIX)')
DEBUG=$($PYTHON -c 'import config; print(config.DEBUG)')
PYVERSION=$($PYTHON -c 'import config; print(config.PYVERSION)')
#export PYTHONPATH="$PYTHONPATH:$(pwd)"

echo "PYTHON: $(which $PYTHON)"
echo "CUDA:   $CUDA"
echo "PREFIX: $PREFIX"
echo "SITE:   $SITE"

if [ $PYVERSION == "False" ]; then
	echo "Python version not high enough need 3.4 or more"
	exit
fi

test config.py -nt mcts/.timestamp && rm -rf mcts/build
test config.py -nt mcts/.timestamp && echo "Removing build directory"

if [ $INSTALL_PYBIND11 == "True" ]; then
	echo "Installing pybind11"
	$PYTHON -m pip install --user pybind11
	pushd pybind11
	mkdir build
	cd build
	cmake -DCMAKE_INSTALL_PREFIX:PATH=$PREFIX .. && \
	make -j8 && make install
	rc=$?
	if [ $rc != 0 ]; then
		echo "Installation failed"
		exit
	fi
	popd
else
	echo "Not installing pybind11"
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
if [ $INSTALL_PYTORCH == "True" ]; then
	echo "Installing pytorch with libtorch"
	pushd pytorch > /dev/null
	git checkout 916a670828bad914907f628e88e6c0ca6bb9b365
	git submodule update --init --recursive
	$PYTHON -m pip install --user pyyaml==3.13
	# typing is not required dependency for python > 3.4
	sed -i '/typing/d' requirements.txt
	$PYTHON -m pip install --user -r requirements.txt

	if [ $CUDA == "False" ]; then
		echo "Building pytorch without cuda"
		exit
		USE_OPENCV=1 \
		BUILD_TORCH=ON \
		CMAKE_PREFIX_PATH="/usr/bin/" \
		LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
		USE_CUDA=0 \
		USE_NNPACK=0 \
		CC=cc \
		CXX=c++ \
		$PYTHON setup.py bdist_wheel && \
		$PYTHON -m pip install --user --upgrade dist/*.whl
		rc=$?
		if [ $rc != 0 ]; then
			echo "Installation failed"
			exit
		fi
	else
		echo "Building pytorch with cuda"
		USE_OPENCV=1 \
		BUILD_TORCH=ON \
		CMAKE_PREFIX_PATH="/usr/bin/" \
		LD_LIBRARY_PATH="$CUDA_PATH/lib64:/usr/local/lib:$LD_LIBRARY_PATH" \
		CUDA_BIN_PATH=$CUDA_PATH/bin \
		CUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH \
		CUDNN_LIB_DIR=$CUDA_PATH/lib64 \
		CUDA_HOST_COMPILER=cc \
		USE_CUDA=1 \
		USE_NNPACK=1 \
		CC=cc \
		CXX=c++ \
		TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" \
		TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
		$PYTHON setup.py bdist_wheel && \
		$PYTHON -m pip install --user --upgrade dist/*.whl
		rc=$?
		if [ $rc != 0 ]; then
			echo "Installation failed"
			exit
		fi
	fi

	popd
else
	echo "Not installing pytorch"
fi

pushd mcts > /dev/null
echo "Installing cmcts module to $PREFIX"
$PYTHON setup.py install --prefix $PREFIX
rc=$?
if [ $rc != 0 ]; then
	echo "Installation failed"
	exit
fi
popd > /dev/null
touch mcts/.timestamp

if [ $DEBUG == "True" ]; then
	echo "Creating missing directories"
fi
if [ ! -d "parameters" ]; then
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

if [ ! -d $PARAM_BEST ]; then
	echo "Creating .param_best file"
	$PYTHON -c 'import src.tools as t; t.init_param_file()'
fi

echo "Installation finnished successfully"
