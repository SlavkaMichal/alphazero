if [ "$#" -eq "0" ]; then
	PYTHON=$(which python)
elif [ "$#" -eq "1" ]; then
	PYTHON=$1
else
	echo "Usage: source $0 [python]"
fi
export PYTHONUSERBASE="$($PYTHON -c 'import general_config as gc; print(gc.PREFIX)')"
export PATH=$PYTHONUSERBASE/bin:$PATH
export LD_LIBRARY_PATH="$PYTHONUSERBASE/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PYTHONPATH="$PYTHONPATH:$($PYTHON -c 'import general_config as gc; print(gc.LOCAL_SITE_PATH)'):$(pwd)"
