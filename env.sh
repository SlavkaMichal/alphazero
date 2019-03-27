export PYTHONUSERBASE="$(python3.6 -c 'import config; print(config.PREFIX)')"
export PATH=$PYTHONUSERBASE/bin:$PATH
export LD_LIBRARY_PATH=$PYTHONUSERBASE/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$PYTHONPATH:$(python3.6 -c 'import config; print(config.LOCAL_SITE_PATH)'):$(pwd)"
