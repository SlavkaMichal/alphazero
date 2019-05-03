import sys
sys.path.append('..')
from general_config import *
from importlib import import_module
import cmcts
import traceback
import pdb
import glob
import os
import numpy as np
import torch
import logging
from tools import rand_uint32
import tools
from datetime import datetime
import re

def eval_models(param_best, param_latest):
    log_file = "{}/eval_{}_{}.log".format(
            LOG_PATH,
            os.path.basename(param_best).replace(".pyt",''),
            re.sub("^.*?_", '', param_latest).replace(".pyt",''))
    logging.basicConfig(format='%(levelname)s: %(message)s',
            filename=log_file,
            level=logging.DEBUG)
    print("Log saved to {}".format(log_file))
    logging.info("########################################")
    logging.info(tools.str_conf())
    logging.info("MCTS from: {}".format(cmcts.__file__))
    print(tools.str_conf())

    config_best = tools.get_param_conf()
    model_module_best = import_module(config_best.MODEL_MODULE)
    model_class_best = getattr(model_module_best, config_best.MODEL_CLASS)
    model_best = model_class_best(config_best)
    model_best.eval()

    config_latest = tools.get_versus_conf()
    model_module_latest = import_module(config_latest.MODEL_MODULE)
    model_class_latest = getattr(model_module_latest, config_latest.MODEL_CLASS)
    model_latest = model_class_latest(config_latest)
    model_latest.eval()

    logging.info("Best model {}".format(param_best))
    logging.info("Latest model {}".format(param_latest))
    if os.path.isfile(param_best) and os.path.isfile(param_latest):
        logging.info("Loading model parameters from {}".format(param_best))
        param_best_loaded = torch.load(param_best)
        logging.info("Loading model parameters from {}".format(param_latest))
        param_latest_loaded = torch.load(param_latest)
    else:
        logging.info("No parameters provided")
        return False

    jit_model_best = "{}/tmp_{}_best.pt".format(
            os.path.dirname(os.path.realpath(__file__)),
            os.path.basename(param_best).replace(".pyt",''))
    jit_model_latest = "{}/tmp_{}_latest.pt".format(
            os.path.dirname(os.path.realpath(__file__)),
            os.path.basename(param_latest).replace(".pyt",''))
    example = torch.rand(1,2,SHAPE,SHAPE)

    model_best.load_state_dict(param_best_loaded['state_dict'])
    if CUDA:
        example.cuda()
        model_best.cuda()
        model_latest.cuda()

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model_best, example)

    logging.info("Saving best model to {}".format(jit_model_best))
    traced_script_module.save(jit_model_best)
    model_latest.load_state_dict(param_latest_loaded['state_dict'])

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model_latest, example)

    logging.info("Saving latest model to {}".format(jit_model_latest))
    traced_script_module.save(jit_model_latest)

    mcts_best = cmcts.mcts(cpuct=config_best.CPUCT)
    mcts_best.set_alpha_default()
    mcts_best.set_threads(THREADS)
    mcts_best.set_params(jit_model_best)
    mcts_best.eps = config_best.EPS

    mcts_latest = cmcts.mcts(cpuct=config_latest.CPUCT)
    mcts_latest.set_alpha_default()
    mcts_latest.set_threads(THREADS)
    mcts_latest.set_params(jit_model_latest)
    mcts_latest.eps = config_latest.EPS

    wins_best   = 0
    wins_latest = 0
    draws       = 0

    logging.info("Running {} games, timeout is set to {}".format(EVAL_GAMES, TIMEOUT_EVAL))
    start_eval = datetime.now()
    logging.info("Starting at {}".format(start_eval))

    # figuring out which player will move first
    tools.make_init_moves(mcts_best, INIT_MOVES)
    tools.make_init_moves(mcts_latest, INIT_MOVES)
    first_player = mcts_best.player
    second_player = 0 if mcts_best.player == 1 else 1
    logging.info("First player is {}".format(first_player))
    mcts_best.clear()
    mcts_latest.clear()

    for i in range(EVAL_GAMES):
        logging.info("Playing game: {}".format(i))
        start = datetime.now()
        tools.make_init_moves(mcts_best, INIT_MOVES)
        tools.make_init_moves(mcts_latest, INIT_MOVES)

        if i%2 == 0:
            logging.info("First player is best")
            try:
                eval_game(mcts_best, config_best.SIMS, mcts_latest, config_latest.SIMS)
            except Exception as e:
                logging.error("Exception raised: {}".format(e.message))
                logging.error("Traceback: {}".format(traceback.format_exc(g)))
                sys.exit(1)
            if mcts_best.winner == first_player:
                logging.info("Winner is best")
                wins_best += 1
            elif mcts_best.winner == second_player:
                logging.info("Winner is latest")
                wins_latest += 1
            else:
                logging.info("draw")
                draws += 1
        else:
            try:
                eval_game(mcts_latest, config_latest.SIMS, mcts_best, config_best.SIMS)
            except Exception as e:
                logging.error("Exception raised: {}".format(e.message))
                logging.error("Traceback: {}".format(traceback.format_exc(g)))
                sys.exit(1)
            if mcts_best.winner == first_player:
                logging.info("Winner is best")
                wins_latest += 1
            elif mcts_best.winner == second_player:
                logging.info("Winner is latest")
                wins_best += 1
            else:
                logging.info("draw")
                draws += 1

        end = datetime.now()
        logging.info("\n{}".format(mcts_best))
        logging.info("Winner is {}".format(mcts_best.winner))
        logging.info("Game took {}".format(end-start))
        logging.info("Score best:{}, latest:{}, draws:{}".format(wins_best, wins_latest, draws))
        try:
            mcts_best.clear()
            mcts_latest.clear()
        except Exception as e:
            logging.error("Exception raised: {}".format(e.message))
            logging.error("Traceback: {}".format(traceback.format_exc(g)))
            sys.exit(1)

        logging.info("Timeout {}s >= {}s".format((end - start_eval).seconds, 60*TIMEOUT_EVAL))
        if (end - start_eval).seconds >= TIMEOUT_EVAL*60:
            logging.info("Timeout expired")
            break
        logging.info("----------------------------------------")

    print("Final result:")
    print("\tBest wins:   {}".format(wins_best))
    print("\tLatest wins: {}".format(wins_latest))
    print("\tDraws:       {}".format(draws))

    logging.info("Total time {}".format(end-start_eval))
    logging.info("Latest win/loos ratio: {}".format(wins_latest/(wins_latest+wins_best)))
    if wins_latest/(wins_latest+wins_best) >= EVAL_TRESHOLD:
        logging.info("Setting new best to {}".format(param_latest))
        tools.set_best(param_latest)
        tools.init_generation()
    else:
        logging.info("Latest model is not good enoug to replace current best")
    logging.info("####################END#################")

    return True

def eval_game(mcts_first, sims_first, mcts_second, sims_second):
    for i in range(SIZE):
        mcts_first.simulate(sims_first)
        pi = mcts_first.get_prob()
        move = pi.argmax()
        mcts_first.make_move(move)
        mcts_second.make_move(move)

        if mcts_first.winner != -1:
            logging.info("Game ended in {} moves".format(i))
            break

        mcts_second.simulate(sims_second)
        pi = mcts_second.get_prob()
        move = pi.argmax()
        mcts_first.make_move(move)
        mcts_second.make_move(move)

        if mcts_first.winner != -1:
            logging.info("Game ended in {} moves".format(i))
            break
    return

if __name__ == "__main__":
    param_best   = tools.get_params()
    param_latest = tools.get_versus()
    if param_best == None:
        print("Parameters were not found")
        sys.exit(1)
    if param_latest == None:
        print("Parameters were not found")
        sys.exit(1)
    print(param_best, param_latest)
    if param_best == param_latest:
        print("Best and latest are the same")
        sys.exit(1)

    rc = True
    try:
        rc = eval_models(param_best, param_latest)
    except Exception as e:
        logging.error("Exception raised: {}".format(e.message))
        logging.error("Traceback: {}".format(traceback.format_exc(g)))
    if rc:
        print("Evluation of params {} and {} finished successfuly"
                .format(param_best, param_latest))
    else:
        log_file = "{}/eval_{}_{}.log".format(LOG_PATH, param_best, re.sub("^.*?_", '', param_latest))
        print("Evaluation failed, check for errors {}"
                .format(log_file))
        sys.exit(1)
