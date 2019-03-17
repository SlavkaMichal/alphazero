export PYTHONUSERBASE=$(python -c 'import config; print(config.PREFIX)')
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$P(python -c 'import config; print(config.LOCAL_SITE_PATH)')
