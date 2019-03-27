import sys
sys.path.append('..')
from config import *
from importlib import import_module
model_module = import_module(MODEL_MODULE)
import cmcts
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

def eval_models(model_class, param_best, param_latest, dry_run=False):
    log_file = "{}/eval_{}_{}.log".format(
            LOG_PATH,
            os.path.basename(param_best).replace(".pyt",''),
            re.sub("^.*?_", '', param_latest).replace(".pyt",''))
    logging.basicConfig(format='%(levelname)s: %(message)s',
            filename=log_file,
            level=logging.DEBUG)
    logging.info("########################################")

    model = model_class()


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

    jit_model_best = "tmp{}_best.pt".format(os.path.basename(param_best).replace(".pyt",''))
    jit_model_latest = "tmp{}_best.pt".format(os.path.basename(param_latest).replace(".pyt",''))
    example = torch.rand(1,2,SHAPE,SHAPE)

    model.load_state_dict(param_best_loaded['state_dict'])
    if CUDA:
        example.cuda()
        model.cuda()

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, example)

    traced_script_module.save(jit_model_best)

    model.load_state_dict(param_latest_loaded['state_dict'])
    if CUDA:
        model.cuda()

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, example)

    traced_script_module.save(jit_model_latest)

    logging.info("MCTS initialised with alpha default, cpuct {}".format(CPUCT))
    mcts_best = cmcts.mcts(seed=rand_uint32(), cpuct=CPUCT)
    mcts_best.set_alpha_default()
    mcts_best.set_params(jit_model_best)

    mcts_latest = cmcts.mcts(seed=rand_uint32(), cpuct=CPUCT)
    mcts_latest.set_alpha_default()
    mcts_latest.set_params(jit_model_latest)

    wins_best   = 0
    wins_latest = 0
    draws       = 0

    logging.info("Running {} games".format(EVAL_GAMES))
    start_eval = datetime.now()
    logging.info("Starting at {}".format(start_eval))
    logging.info("Best playing {} games as first player".format(EVAL_GAMES//2))

    # figuring out which player will move first
    tools.make_init_moves(mcts_best, mcts_latest)
    first_player = mcts_best.player
    second_player = 0 if mcts_best.player == 1 else 1
    logging.info("Current best player playing as {}".format(first_player))
    logging.info("Latest model playing as {}".format(second_player))
    mcts_best.clear()
    mcts_latest.clear()

    for i in range(EVAL_GAMES//2):
        logging.info("Playing game: {}".format(i))
        start = datetime.now()
        tools.make_init_moves(mcts_best, mcts_latest)
        print(mcts_best)

        eval_game(mcts_best, mcts_latest)
        logging.info("Winner is {}".format(mcts_best.winner))
        if mcts_best.winner == first_player:
            wins_best += 1
        elif mcts_best.winner == second_player:
            wins_latest += 1
        else:
            draws += 1
        logging.info("Game took {}".format(datetime.now()-start))
        logging.info("Score best:{}, latest:{}, draws:{}".format(wins_best, wins_latest, draws))

        print(mcts_best)
        mcts_best.clear()
        mcts_latest.clear()

    logging.info("First round of games:")
    logging.info("\tBest wins:   {}".format(wins_best))
    logging.info("\tLatest wins: {}".format(wins_latest))
    logging.info("\tDraws:       {}".format(draws))

    logging.info("Latest playing {} games as first player".format(EVAL_GAMES//2))
    # swaping players
    logging.info("Current best player playing as {}".format(first_player))
    logging.info("Latest model playing as {}".format(second_player))

    for i in range(EVAL_GAMES//2):
        logging.info("Playing game: {}".format(i))
        start = datetime.now()
        tools.make_init_moves(mcts_best, mcts_latest)
        print(mcts_best)

        # here is the difference from previous loop
        eval_game(mcts_latest, mcts_best)
        logging.info("Winner is {}".format(mcts_best.winner))
        if mcts_best.winner == first_player:
            wins_latest += 1
        elif mcts_best.winner == second_player:
            wins_best += 1
        else:
            draws += 1
        logging.info("Game took {}".format(datetime.now()-start))
        logging.info("Score best:{}, latest:{}, draws:{}".format(wins_best, wins_latest, draws))

        print(mcts_best)
        mcts_best.clear()
        mcts_latest.clear()

    logging.info("Evaluation took {}".format(datetime.now()-start_eval))
    logging.info("Final result:")
    logging.info("\tBest wins:   {}".format(wins_best))
    logging.info("\tLatest wins: {}".format(wins_latest))
    logging.info("\tDraws:       {}".format(draws))

    logging.info("Latest win/loos ratio: {}".format(wins_latest/(wins_latest+wins_best)))
    if wins_latest/(wins_latest+wins_best) > 0.54:
        logging.info("Setting new best to {}".format(param_latest))
        if not dry_run:
            tools.set_best(param_latest)
    else:
        logging.info("Latest model is not good enoug to replace current best")
    logging.info("####################END#################")

    return True

def eval_game(mcts_first, mcts_second):
    for i in range(SIZE):
        mcts_first.simulate(SIMS)
        pi = mcts_first.get_prob()
        move = np.random.choice(pi.size, p=pi)
        mcts_first.make_move(move)
        mcts_second.make_move(move)

        if mcts_first.winner != -1:
            break

        mcts_second.simulate(SIMS)
        pi = mcts_second.get_prob()
        move = np.random.choice(pi.size, p=pi)
        mcts_first.make_move(move)
        mcts_second.make_move(move)

        if mcts_first.winner != -1:
            break

    return

if __name__ == "__main__":
    param_best, param_latest = tools.info_eval()
    if param_best == None:
        print("Best model {} is also the latest".format(param_latest))
        sys.exit(1)
    model_class = getattr(model_module, MODEL_CLASS)
    if eval_models(model_class, param_best, param_latest, True):
        print("Evluation of params {} and {} finished successfuly"
                .format(param_best, param_latest))
    else:
        log_file = "{}/eval_{}_{}.log".format(LOG_PATH, param_best, re.sub("^.*?_", '', param_latest))
        print("Evaluation failed, check for errors {}"
                .format(log_file))
        sys.exit(1)
