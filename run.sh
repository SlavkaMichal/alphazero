#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

HELP="Usage: bash $0 [ACTION] [OPTION] [--python=[PYTHON INTERPRETER]\n\n
 ACTIONS\n
  -s, --self-play\tgenerate data with self play\n
  -t, --train\ttrain neural network\n
  -e, --eval\tevaluate current best nn against latest\n\n
  -g  --generation\t\tnew generation\n
  -q  --set-best\t\tset best parameters to latest (can be combined with -v parameter)\n
  -r, --restore\t\t\trestore last configureation file\n\n
 OPTIONS\n
  -o, --param=[PARAM FILE]\tparameters that will be used for each action,\n
                          \t\t\t\tif -e is specified -v is required\n
  -v, --versus=[PARAM FILE]\tmodel parameters that will be used for evaluation\n
                           \t\t\t\tagainst other supplied parameters\n
  -a  --conf-param=[PARAM FILE]\tspecify configuration file\n
  -b  --conf-versus=[PARAM FILE]\tspecify configuration file for oponent\n
  -n  --dry-run\t\t\tdo not create new generation or set new best parameters\n
  -d, --data=[DATA LIST]\t\tcoma-separated list of data files, NO SPACES can be used\n
  -c, --config=[CONFIG NAME]\tload configuration file and architecture,\n
                            \t\t\t\tif in combination with --train original configuration will be restored\n
  -i, --sequence=[SEQ NUMBER]\tif running multiple scripts at once it's good to add sequence number\n
  -h, --help\t\t\tprint this\n
  -p, --python=[PYTHON]\t\tpath to python interpret\n\n
  Configure scripts in file $(pwd)/config.py
"

PYTHON=$(which python)
ACTION="n"
CONFIG=""
source env.sh $PYTHON

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
		echo "SEQUENCE=$SEQUENCE"
		export SEQUENCE
		shift
		shift
		;;
	-o|--param)
		PARAM="$2"
		echo "PARAM=$PARAM"
		export PARAM
		shift
		shift
		;;
	-v|--versus)
		VERSUS="$2"
		echo "VERSUS=$VERSUS"
		export VERSUS
		shift
		shift
		;;
	-a|--conf-param)
		CONF_PARAM="$2"
		echo "CONF_PARAM=$CONF_PARAM"
		export CONF_PARAM
		shift
		shift
		;;
	-b|--conf-versus)
		CONF_VERSUS="$2"
		echo "CONF_VERSUS=$CONF_VERSUS"
		export CONF_VERSUS
		shift
		shift
		;;
	-d|--data)
		DATA="$2"
		echo "DATA=$DATA"
		export DATA
		shift
		shift
		;;
	-c|--config)
		CONFIG="$2"
		echo "CONFIG=$CONFIG"
		$PYTHON -c "import src.tools as t; t.config_save('tmp.py'); t.config_load('$CONFIG')"
		if [ $ACTION == "n" ]; then
			echo "Config succesfully loaded"
			echo "To restore original config run '$0 -r | --resotre'"
			exit
		fi
		export CONFIG
		shift
		shift
		;;
	-r|--restore)
		if [ -f "config/tmp_conf.py" ]; then
			$PYTHON -c "import src.tools as t; t.config_load('config/tmp_conf.py')"
			rm -v "config/tmp_conf.py"
		else
			echo "Temporary configuration file does not exists"
		fi
		shift
		exit
		;;
	-q| --set-best)
		ACTION="q"
		shift
		;;
	-g|--generation)
		$PYTHON -c "import src.tools as t; t.init_generation()"
		shift
		exit
		;;
	-n|--dry-run)
		DRY_RUN="true"
		export DRY_RUN
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

if [ $ACTION == "n" ]; then
	echo -e "Error: No action giveni\n"
	echo -e $HELP
	exit
elif [ $ACTION == "q" ]; then
	$PYTHON -c "import src.tools as t; t.set_best(t.get_versus())"
elif [ $ACTION == "s" ]; then
	echo "running $PYTHON self-play.py"
	$PYTHON src/self_play.py
	# restore cofig if a different one was used
	if [ "$CONFIG" != "" ]; then
		$PYTHON -c "import src.tools as t; t.config_load('config/tmp_conf.py')"
		rm 'config/tmp_conf.py'
	fi
elif [ $ACTION == "e" ]; then
	echo "running $PYTHON eval.py"
	# restore cofig if a different one was used
	$PYTHON "src/eval.py" && echo "Success"
	if [ "$CONFIG" != "" ]; then
		$PYTHON -c "import src.tools as t; t.config_load('config/tmp_conf.py')"
		rm 'config/tmp_conf.py'
	fi
elif [ $ACTION == "t" ]; then
	echo "running $PYTHON train.py"
	# restore cofig if a different one was used
	$PYTHON src/train.py
	if [ "$CONFIG" != "" ]; then
		$PYTHON -c "import src.tools as t; t.config_load('config/tmp_conf.py')"
		rm 'config/tmp_conf.py'
	fi
fi
