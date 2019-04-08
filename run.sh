#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

HELP="Usage: bash $0 ACTION [--python=[PYTHON INTERPRETER]\n\n
  -s, --self-play\tgenerate data with self play\n
  -t, --train\ttrain neural network\n
  -e, --eval\tevaluate current best nn against latest\n
  -h, --help\tprint this\n
  -p, --python=[PYTHON]\tpath to python interpret\n\n
  Configure scripts in file $(pwd)/config.py
"

PYTHON=$(which python)
ACTION="n"
SEQUENCE="0"

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
	-p|--python)
		PYTHON="$2"
		shift
		shift
		;;
	*)
esac
done

if [ "x$SEQUENCE" != "x" ]; then
	echo "SEQUENCE=$SEQUENCE"
	export SEQUENCE
fi
source env.sh $PYTHON

if [ $ACTION == "n" ]; then
	echo -e "Error: No action giveni\n"
	echo -e $HELP
	exit
elif [ $ACTION == "s" ]; then
	echo "running $PYTHON self-play.py"
	$PYTHON src/self_play.py
elif [ $ACTION == "t" ]; then
	echo "running $PYTHON train.py"
	$PYTHON src/train.py
elif [ $ACTION == "e" ]; then
	echo "running $PYTHON eval.py"
	$PYTHON src/eval.py
fi
