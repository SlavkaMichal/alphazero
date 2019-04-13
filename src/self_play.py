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

def self_play_iteration(model_class, param_file=None, data_file=None):
    logging.basicConfig(format='%(levelname)s: %(message)s',
            filename="{}/self-play_{}.log".format(LOG_PATH, os.path.basename(data_file).replace(".pyt",'')),
            level=logging.DEBUG)
    print("Log saved to: {}/self-play_{}.log".format(LOG_PATH, os.path.basename(data_file).replace(".pyt",'')))
    logging.info("########################################")
    if DEBUG:
        print(tools.self_play_config())
    logging.info(tools.self_play_config())

    model = model_class()

    if os.path.isfile(param_file):
        logging.info("Loading model parameters from {}".format(param_file))
        params = torch.load(param_file)
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
            os.path.basename(data_file).replace("{MODEL_CLASS}_", ''),
            os.path.basename(param_file).replace(".pyt",''))
    example = torch.rand(1,2,SHAPE,SHAPE)

    if CUDA:
        example.cuda()
        model.cuda()

    model.eval()

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, example)

    logging.info("Saving traced script to {}".format(jit_model_name))
    traced_script_module.save(jit_model_name)

    logging.info("MCTS initialised with alpha default, cpuct {}".format(CPUCT))
    mcts0 = cmcts.mcts(seed=rand_uint32(), cpuct=CPUCT)
    mcts0.set_alpha_default()
    mcts0.set_threads(THREADS)
    mcts0.set_params(jit_model_name)

    mcts1 = cmcts.mcts(seed=rand_uint32(), cpuct=CPUCT)
    mcts1.set_alpha_default()
    mcts1.set_threads(THREADS)
    mcts1.set_params(jit_model_name)

    # dtype should be always dtype of input tensor
    data = []

    i = 0
    logging.info("Generating {} training samples".format(TRAIN_SAMPLES))
    start_gen = datetime.now()
    logging.info("Starting at {}".format(start_gen))
    while len(data) < TRAIN_SAMPLES:
        logging.info("Playing game: {}".format(i))
        start = datetime.now()
        tools.make_init_moves(mcts0, mcts1)

        game_data = self_play_game(mcts0, mcts1)
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

def self_play_game(mcts0, mcts1):
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
        mcts0.simulate(SIMS)
        pi = mcts0.get_prob()
        board = mcts0.get_board()
        data0.append(np.array((board, pi, -1), dtype=dt))
        move = np.random.choice(pi.size, p=pi)
        #logging.info("Player {}".format(mcts0.player))
        #logging.info("Move {}".format(move))
        #logging.info("Board {}".format(tools.repr_board(board)))
        #logging.info("Pi {}".format(tools.repr_pi(pi)))
        mcts0.make_move(move)
        mcts1.make_move(move)

        if mcts0.winner != -1:
            break

        mcts1.simulate(SIMS)
        pi = mcts1.get_prob()
        board = mcts1.get_board()
        data1.append(np.array((board, pi, -1), dtype=dt))
        move = np.random.choice(pi.size, p=pi)
        #logging.info("Player {}".format(mcts0.player))
        #logging.info("Move {}".format(move))
        #logging.info("Board {}".format(tools.repr_board(board)))
        #logging.info("Pi {}".format(tools.repr_pi(pi)))
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

    #data = []
    #for i in range(len(data1)):
    #    data.append(data0[i])
    #    data.append(data1[i])
    #if len(data0) > len(data1):
    #    data.append(data0[-1])

    #data = np.stack(data)
    return data

if __name__ == "__main__":
    param_file, data_file = tools.info_selfplay()
    model_class = getattr(model_module, MODEL_CLASS)
    if self_play_iteration(model_class, param_file, data_file):
        print("Data were saved to file {}.npy".format(data_file))
    else:
        print("Generating data failed, check for errors {}/self-play_{}.log".format(LOG_PATH, os.path.basename(data_file).replace(".pyt",'')))
