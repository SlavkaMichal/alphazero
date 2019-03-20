import sys
sys.path.append('..')
from config import *
import site
site.addsitedir(LOCAL_SITE_PATH)
from model import simplerNN
import cmcts
import glob
import os
import numpy as np
import torch
import logging
from tools import rand_uint32
import tools
from datetime import datetime

def self_play_iteration(model_class):
    """
        model name:
          [class name of model][board size]_[generation]
        parameters name:
          [class name of model][board size]_[generation].pt
        datafile name:
          [class name of model][board size]_[iteration]_[generation].npy
    """
    model_name = "{}{}".format(model_class.__name__,SHAPE)
    # read info
    iteration, generation, best = tools.info_read()
    iteration = iteration + 1

    logging.basicConfig(format='%(levelname)s: %(message)s', filename="../logs/self-play{:03d}.log".format(iteration), level=logging.DEBUG)
    logging.info("Iteration number {}".format(iteration))
    logging.info("Model generation: {}".format(best))

    model = model_class()

    if best == -1 and generation == -1:
        # if there are no models save initialization parameters
        # increment generation to 0 and save current model parameters
        generation = tools.info_generation()
        # set zero generation as the best
        tools.info_best(generation)
        param_file = "../model/{}_{:03d}.pt".format(model_name, generation)
        logging.info("No model parameters set")
        logging.info("Saving new model parameters to {}".format(param_file))
        torch.save({
            'state_dict' : model.state_dict(),
            'model_name' : "{}_{}".format(model_name, generation),
            'generation' : 0
            }, param_file)
    else:
        if best == -1:
            best = generation
        param_file = "../model/{}_{}.pt".format(model_name, best)
        logging.info("Loading model parameters from {}".format(param_file))
        if (not os.path.isfile(param_file)):
            logging.info("No file with filename {} found".format(param_file))
            logging.info("Saving new model parameters to {}".format(param_file))
            torch.save({
                'state_dict' : model.state_dict(),
                'model_name' : "{}_{}".format(model_name, generation),
                'generation' : 0
                }, param_file)
            tools.info_set(0, 0, 0)
        else:
            params = torch.load(param_file)
            model.load_state_dict(prams['state_dict'])

    logging.info("MCTS initialised with alpha {}, cpuct {}")

    if CUDA:
        example = torch.rand(1,2,SHAPE,SHAPE).cuda()
        model.cuda()
    else:
        example = torch.rand(1,2,SHAPE,SHAPE)

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, example)

    traced_script_module.save("{}.pt".format(model_name))
    return

    mcts0 = cmcts.mcts(seed=rand_uint32(), cpuct=CPUCT)
    mcts0.set_predictor("{}.pt".format(model_name))
    mcts1 = cmcts.mcts(seed=rand_uint32(), cpuct=CPUCT)
    mcts1.set_predictor("{}.pt".format(model_name))
    if 'ALPHA' in globals():
        mcts0.set_alpha(ALPHA)
        mcts1.set_alpha(ALPHA)
    else:
        mcts0.set_alpha()
        mcts1.set_alpha()

    # dtype should be always dtype of input tensor
    data = []

    i = 0
    while len(data) < TRAIN_SAMPLES:
        logging.info("Playing game: {}".format(i))
        start = datetime.now()
        game_data = self_play_game(mcts0, mcts1)
        logging.info("Game ended in {} moves".format(game_data))
        logging.info("Game took {}".format(datetime.now()-start))
        data.extend(game_data)
        mcts0.clear()
        mcts1.clear()
        i = i + 1
        # TODO this only plays one game and it's taking fucking long

    npdata = np.stack(data)
    np.random.shuffle(npdata)
    data_file  = "../data/{}_{:03d}".format(model_name, iteration)
    logging.info("Saving data to: {}".format(data_file))
    tools.info_iteration()
    np.save(data_file, npdata)

    os.remove("{}.pt".format(model_name))
    return

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

    dt = np.dtype([('board', 'f4', (2,SHAPE,SHAPE)), ('pi', 'f4', (SIZE,)), ('r', np.int64)])

    for i in range(SIZE//2):
        mcts0.simulate(SIMS)
        pi = mcts0.get_prob()
        board = mcts0.get_board()
        data0.append(np.array((board, pi, -1), dtype=dt))
        move = np.random.choice(pi.size, p=pi)
        mcts0.make_move(move)
        mcts1.make_move(move)

        if mcts0.winner != -1:
            break

        mcts1.simulate(SIMS)
        pi = mcts1.get_prob()
        board = mcts1.get_board()
        data1.append(np.array((board, pi, -1), dtype=dt))
        move = np.random.choice(pi.size, p=pi)
        mcts0.make_move(move)
        mcts1.make_move(move)

        if mcts0.winner != -1:
            break

    length = len(data0)
    data0.extend(data1)
    data = np.stack(data0)
    data_view = np.array_split(data,[length])

    if mcts0.winner == 0:
        data_view[0]['r'] = 1
    elif mcts0.winner == 0:
        data_view[1]['r'] = 1
    else:
        # draw
        data['r'] = 1e-3

    return data

def model_wraper(board, model):
    with torch.no_grad():
        tboard = torch.as_tensor(board)
        v, p = model(tboard.reshape(1,2,SHAPE,SHAPE))

    return v[0].item(), p[0].numpy()


if __name__ == "__main__":
    self_play_iteration(simplerNN)
    pass
