import sys
sys.path.append('..')
from general_config import *
from importlib import import_module
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

def self_play_iteration(param_file=None, data_file=None):
    logging.basicConfig(format='%(levelname)s: %(message)s',
            filename="{}/self-play_{}.log".format(LOG_PATH, os.path.basename(data_file).replace(".pyt",'')),
            level=logging.DEBUG)
    print("Log saved to: {}/self-play_{}.log".format(LOG_PATH, os.path.basename(data_file).replace(".pyt",'')))
    logging.info("########################################")
    if DEBUG:
        print(tools.str_conf())
    logging.info(tools.str_conf())

    config = tools.get_param_conf()
    model_module = import_module(config.MODEL_MODULE)
    model_class = getattr(model_module, config.MODEL_CLASS)
    model = model_class(config)

    if os.path.isfile(param_file):
        logging.info("Loading model parameters from {}".format(param_file))
        params = torch.load(param_file, map_location='cpu')
        try:
            model.load_state_dict(params['state_dict'])
        except RuntimeError as e:
            logging.info("Could not load parametrs")
            logging.info(e)
            logging.info("Saving new model parameters to {}".format(param_file))
            torch.save({
                'state_dict' : model.state_dict(),
                }, param_file)
    else:
        logging.info("No parameters provided")
        logging.info("Saving new model parameters to {}".format(param_file))
        torch.save({
            'state_dict' : model.state_dict(),
            }, param_file)

    jit_model_name = "{}/tmp_{}{}.pt".format(
            os.path.dirname(os.path.realpath(__file__)),
            os.path.basename(data_file).replace("{config.MODEL_CLASS}_", ''),
            os.path.basename(param_file).replace(".pyt",''))
    example = torch.rand(1, 2, SHAPE, SHAPE)

    if CUDA:
        example.cuda()
        model.cuda()

    model.eval()

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, example)

    logging.info("Saving traced script to {}".format(jit_model_name))
    traced_script_module.save(jit_model_name)

    logging.info("MCTS initialised with alpha default, cpuct {}".format(config.CPUCT))
    mcts0 = cmcts.mcts(cpuct=config.CPUCT)
    mcts0.set_alpha_default()
    mcts0.set_threads(THREADS)
    mcts0.set_params(jit_model_name)
    mcts0.eps = config.EPS

    mcts1 = cmcts.mcts(cpuct=config.CPUCT)
    mcts1.set_alpha_default()
    mcts1.set_threads(THREADS)
    mcts1.set_params(jit_model_name)
    mcts1.eps = config.EPS

    # dtype should be always dtype of input tensor
    data = []

    i = 0
    logging.info("Generating {} training samples".format(TRAIN_SAMPLES))
    start_gen = datetime.now()
    logging.info("Starting at {}".format(start_gen))
    while len(data) < TRAIN_SAMPLES:
        logging.info("Playing game: {}".format(i))
        start = datetime.now()
        tools.make_init_moves(mcts0, INIT_MOVES)
        tools.make_init_moves(mcts1, INIT_MOVES)

        try:
            game_data = self_play_game(mcts0, mcts1, config.SIMS, config.TAU)
        except Exception as e:
            logging.error("Exception raised: {}".format(e.message))
            logging.error("Traceback: {}".format(traceback.format_exc(g)))
            sys.exit(1)
        data.extend(game_data)
        mcts0.clear()
        mcts1.clear()
        end = datetime.now()
        logging.info("Game ended in {} moves".format(len(game_data)))
        logging.info("Game took {}".format(end-start))
        i = i + 1
        if i%10  == 9:
            logging.info("Creating checkpoint with {} games".format(i))
            npdata = np.stack(data)
            logging.info("Played in total {} ({} moves, {}s per move)".format(npdata.shape[0],npdata.shape[0]/i,(end-start_gen)/i))
            np.save(data_file, npdata)
            del npdata

        logging.info("Timeout {}s >= {}s".format((end - start_gen).seconds, 60*TIMEOUT_SELF_PLAY))
        if (end - start_gen).seconds >= TIMEOUT_SELF_PLAY * 60:
            logging.info("Timeout expired")
            break
        # TODO this only plays one game and it's taking fucking long

    npdata = np.stack(data)
    end = datetime.now()
    logging.info("Finnished at {}".format(end))
    logging.info("Total time {}".format(end-start_gen))
    logging.info("Played {} games".format(i))
    logging.info("Played in total {} ({} moves, {}s per move)".format(npdata.shape[0],npdata.shape[0]/i,(end-start_gen)/i))

    logging.info("Saving data to: {}.npy".format(data_file))
    np.save(data_file, npdata)
    os.remove(jit_model_name)

    logging.info("####################END#################")
    return True

def self_play_game(mcts0, mcts1, sims, tau):
    """ plays one game
        for sefl play model0 and model1 should be the same
        also mcts0 and mcts1 can be the same

        model0 - fist model (the one that is on move)
        model1 - second model
        mcts0  - tree search object for fist model
        mcts1  - tree search object for second model
        board  - game board

        it is intended to run this method with clear initial state
        cleared mcts's and zeroed board

        returns training data
    """
    data0 = []
    data1 = []

    dt = np.dtype([('board', 'f4', (2,SHAPE,SHAPE)), ('pi', 'f4', (SIZE,)), ('r', 'f4')])

    for i in range(SIZE):
        mcts0.simulate(sims)
        pi = mcts0.get_prob()
        board = mcts0.get_board()
        if (i > tau):
            move = pi.argmax()
            pi[:] = 0
            pi[move] = 1
        else:
            move = np.random.choice(pi.size, p=pi)
        data0.append(np.array((board, pi, -1), dtype=dt))
        mcts0.make_move(move)
        mcts1.make_move(move)

        if mcts0.winner != -1:
            break

        mcts1.simulate(sims)
        pi = mcts1.get_prob()
        board = mcts1.get_board()
        if (i > tau):
            move = pi.argmax()
            pi[:] = 0
            pi[move] = 1
        else:
            move = np.random.choice(pi.size, p=pi)
        data1.append(np.array((board, pi, -1), dtype=dt))
        mcts0.make_move(move)
        mcts1.make_move(move)

        if mcts0.winner != -1:
            break

    length = len(data0)
    data0.extend(data1)
    data = np.stack(data0)
    data_view = np.array_split(data,[length])

    logging.info(mcts0)
    if mcts0.winner == 0.:
        data_view[0]['r'] = 1.
    elif mcts0.winner == 1.:
        data_view[1]['r'] = 1.
    else:
        # draw
        data['r'] = 1e-3

    return data

if __name__ == "__main__":
    param_file = tools.get_params()
    if param_file is None:
        param_file = tools.get_new_params()
    data_file  = tools.get_new_data()
    if self_play_iteration(param_file, data_file):
        print("Data were saved to file {}.npy".format(data_file))
    else:
        print("Generating data failed, check for errors {}/self-play_{}.log".format(LOG_PATH, os.path.basename(data_file).replace(".pyt",'')))
