#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

HELP="Usage: bash $0 [ACTION] [OPTION] [--python=[PYTHON INTERPRETER]\n\n
 ACTIONS\n
  -s, --self-play\tgenerate data with self play\n
  -t, --train\ttrain neural network\n
  -e, --eval\tevaluate current best nn against latest\n\n
 OPTIONS\n
  -p, --param=[PARAM FILE]\tparameters that will be used for each action,\n
                          \t\t\t\tif -e is specified -v is required\n\n
  -v, --versus=[PARAM FILE]\tmodel parameters that will be used for evaluation
                           \t\t\t\tagainst other supplied parameters\n
  -d, --data=[DATA LIST]\t\tcoma-separated list of data files, NO SPACES can be used\n
  -c, --config=[CONFIG NAME]\tload configuration file and architecture,\n
                            \t\t\t\tif in combination with --train original configuration will be restored\n\n
  -r, --restore\trestore last configureation file\n
  -i, --sequence=[SEQ NUMBER]\tif running multiple scripts at once it's good to add sequence number\n
  -h, --help\t\t\tprint this\n
  -p, --python=[PYTHON]\t\tpath to python interpret\n\n
  Configure scripts in file $(pwd)/config.py
"

PYTHON=$(which python)
ACTION="n"
SEQUENCE="0"
CONFIG=""
PARAM=""
DATA_FILES=""
VERSUS=""
RESTORE="false"

while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
	-h|--help)
		echo -e $HELP
		exit
		;;
	-s|--self-play)
		ACTION="s"
		shift
		;;
	-t|--train)
		ACTION="t"
		shift
		;;
	-e|--eval)
		ACTION="e"
		shift
		;;
	-i|--sequence)
		SEQUENCE="$2"
		shift
		shift
		;;
	-p|--param)
		PARAM="$2"
		shift
		shift
		;;
	-v|--versus)
		VERSUS="$2"
		shift
		shift
		;;
	-d|--data)
		DATA_FILES="$2"
		shift
		shift
		;;
	-c|--config)
		CONFIG="$2"
		shift
		shift
		;;
	-r|--restore)
		RESTORE="true"
		shift
		;;
	-p|--python)
		PYTHON="$2"
		shift
		shift
		;;
	*)
esac
done

if [ "$RESTORE" == "true" ]; then
	if [ -f "config/tmp_conf.py" ]; then
		$PYTHON -c "import src.tools as t; t.config_load('config/tmp_conf.py')"
		rm -v "config/tmp_conf.py"
	else
		echo "Temporary configuration file does not exists"
	fi
	exit
fi
if [ "$SEQUENCE" != "" ]; then
	echo "SEQUENCE=$SEQUENCE"
	export SEQUENCE
fi
if [ "$CONFIG" != "" ]; then
	echo "CONFIG=$CONFIG"
	$PYTHON -c "import src.tools as t; t.config_save('tmp.py'); t.config_load('$CONFIG')"
	if [ $ACTION == "n" ]; then
		echo "Config succesfully loaded"
		echo "To restore original config run '$0 -r | --resotre'"
		exit
	fi
	export CONFIG
fi
if [ "$PARAM" != "" ]; then
	echo "PARAM=$PARAM"
	export PARAM
fi
if [ "$VERSUS" != "" ]; then
	echo "VERSUS=$CONFIG"
	export VERSUS
fi
if [ "$DATA_FILES" != "" ]; then
	echo "DATA_FILES=$CONFIG"
	export DATA_FILES
fi

source env.sh $PYTHON

if [ $ACTION == "n" ]; then
	echo -e "Error: No action giveni\n"
	echo -e $HELP
	exit
elif [ $ACTION == "s" ]; then
	echo "running $PYTHON self-play.py"
	$PYTHON src/self_play.py
	if [ "$CONFIG" != "" ]; then
		$PYTHON -c "import src.tools as t; t.config_load('config/tmp_conf.py')"
		rm 'config/tmp_conf.py'
	fi
elif [ $ACTION == "e" ]; then
	echo "running $PYTHON eval.py"
	$PYTHON "src/eval.py" && echo "Success"
	if [ "$CONFIG" != "" ]; then
		$PYTHON -c "import src.tools as t; t.config_load('config/tmp_conf.py')"
		rm 'config/tmp_conf.py'
	fi
elif [ $ACTION == "t" ]; then
	echo "running $PYTHON train.py"
	$PYTHON src/train.py
	if [ "$CONFIG" != "" ]; then
		$PYTHON -c "import src.tools as t; t.config_load('config/tmp_conf.py')"
		rm 'config/tmp_conf.py'
	fi
fi
