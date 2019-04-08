import sys
sys.path.append('..')
from config import *
import numpy as np
import json
from glob import glob
import os
from datetime import datetime
import cmcts
import torch

info_file = "../.info"
best_file = "../.best"

def rand_uint32():
    return np.random.randint(np.iinfo(np.int32).max, dtype=np.uint32).item()

def repr_board(board):
    b = "\n"
    for i in range(SHAPE):
        for j in range(SHAPE):
            if board[0,i,j] == 1:
                b += 'x '
            elif board[1,i,j] == 1:
                b += 'o '
            else:
                b += '_ '
        b += "\n"
    return b

def repr_pi(pi):
    p = "\n"
    for i in range(SHAPE):
        for j in range(SHAPE):
            p += "{0:.3f} ".format(pi[i*SHAPE+j])
        p += "\n"
    return p

def make_init_moves(mcts0, mcts1):
    mcts0.make_move(85)
    mcts1.make_move(85)

    mcts0.make_move(98)
    mcts1.make_move(98)

    mcts0.make_move(86)
    mcts1.make_move(86)

    mcts0.make_move(84)
    mcts1.make_move(84)

    mcts0.make_move(112)
    mcts1.make_move(112)

    mcts0.make_move(70)
    mcts1.make_move(70)
    return

def get_unique(data):
    board, idc, inv, cnt = np.unique(data['board'], axis=0,
            return_index=True,
            return_inverse=True,
            return_counts=True)
    #dt = np.dtype([('board', 'f4', (2,SHAPE,SHAPE)), ('pi', 'f4', (SIZE,)), ('r', 'f4')])
    dt = data.dtype
    result = []
    for i in range(idc.size):
        if cnt[i] > 1:
            r  = np.average(data['r'][np.where(inv == i)])
            pi = np.average(data['pi'][np.where(inv == i)], axis=0)
        else:
            r  = data['r'][idc[i]]
            pi = data['pi'][idc[i]]
        result.append(np.array((board[i], pi, r), dtype=dt))

    return np.stack(result)

def get_rotations(data):
    rotations = []
    dt = data.dtype
    shape = data['board'].shape[-1]

    for d in data:
        board = d['board']
        pi = d['pi'].reshape(shape,shape)
        r = d['r']

        for r in range(7):
            if r == 3:
                board = np.flip(board, 1)
                pi    = np.flip(pi, 0)
            else:
                board = np.rot90(board, axes=(1,2))
                pi    = np.rot90(pi)

            rotations.append(np.array((board, pi.reshape(-1), r), dtype=dt))

    return np.concatenate([data, np.stack(rotations)])

def info_train():
    best = get_best()

    if best is None:
        best = get_latest()
        if best is None:
            raise RuntimeError("No model to train")
        set_best(best)

    new_param_file = "{}/{}{}_{}.pyt".format(
            PARAM_PATH, MODEL_CLASS, SHAPE, datetime.now().strftime("%m%d_%H-%M-%S"))
    data_files = glob("{}/{}{}_*.npy".format(
        DATA_PATH, MODEL_CLASS, SHAPE))

    data_files.sort()
    #window = min(WINDOW[0]+(len(data_files)-4)//WINDOW[2], WINDOW[1])
    #window *= PROC_NUM
    #if len(data_files) - window > 0:
    #    for f in data_files[0:len(data_files) - window]:
    #        logging.info("Removing file {}".format(f))
    #        os.remove(f)

    data_files = [ os.path.realpath(df) for df in data_files ]

    return best, new_param_file, data_files

def info_generate():
    """ returns best params and new data file name
        must return valid values!!
    """
    best = get_best()

    if 'SEQUENCE' in os.environ:
        seq = os.environ['SEQUENCE']
    else:
        seq = 0

    if best is None:
        best = get_latest()
    if best is None:
        best = "{}/{}{}_{}.pyt".format(
                PARAM_PATH, MODEL_CLASS, SHAPE, datetime.now().strftime("%m%d_%H-%M-%S"))

    file_name = "{}/{}{}_{}s{}".format(
                DATA_PATH, MODEL_CLASS, SHAPE, datetime.now().strftime("%m%d_%H-%M-%S"), seq)

    return best, file_name

def info_eval():
    best = get_best()
    latest = get_latest()
    if best == latest:
        return None, latest
    return best, latest

def get_latest():
    param_files = glob("{}/{}{}_*.pyt".format(
        PARAM_PATH, MODEL_CLASS, SHAPE))
    if len(param_files) == 0:
        return None

    if not os.path.isabs(param_files[-1]):
        return "{}/{}".format(PARAM_PATH, param_files[-1])
    return param_files[-1]


def get_best():
    with open(PARAM_BEST, "r+") as fp:
        try:
            content = json.load(fp)
        except ValueError:
            path = get_latest()
            if path is None:
                return None
            set_best(path)
            return path

        if not os.path.isfile(content['path']):
            raise RuntimeError("File {} does not exst".format(path))
        return content['path']

def set_best(path):
    with open(PARAM_BEST, "w") as fp:
        if not os.path.isabs(path):
            path = "{}/{}".format(PARAM_PATH, path)
        if not os.path.isfile(path):
            raise RuntimeError("File {} does not exst".format(path))

        path = os.path.normpath(path)
        content = { "path" : path }
        json.dump(content, fp)

def init_param_file():
    open(PARAM_BEST, 'a').close

def self_play_config():
    s = ""
    s += "Number of training samples to generate: {}\n".format(TRAIN_SAMPLES)
    s += "Timeout for self play: {} min\n".format(TIMEOUT_SELF_PLAY)
    s += "Data destination: {}\n".format(DATA_PATH)
    s += "Threads: {}\n".format(DATA_PATH)
    s += "Parameters loaded from: {}\n".format(PARAM_PATH)
    s += "Model from: {}.{}\n".format(MODEL_MODULE, MODEL_CLASS)
    s += "Torch from: {}\n".format(torch.__file__)
    s += "MCTS from: {}\n".format(cmcts.__file__)
    s += "MCTS number of threads: {}\n".format(THREADS)
    return s

def train_config():
    s = ""
    s += "Data source: {}".format(DATA_PATH)
    s += "Parameters loaded from: {}".format(PARAM_PATH)
    s += "Model from: {}.{}".format(MODEL_MODULE, MODEL_CLASS)
    s += "Torch from: {}".format(torch.__file__)
    s += "Learning rate: {}".format(LR)
    s += "Epochs: {}".format(EPOCHS)
    s += "Batch size: {}".format(BATCH_SIZE)
    s += "Starting window size: {}".format(WINDOW[0])
    s += "Max window size: {}".format(WINDOW[1])
    s += "Window size incremented every {} generation".format(WINDOW[2])
    return s

def eval_config():
    s = ""
    s += "Parameters loaded from: {}".format(PARAM_PATH)
    s += "Model from: {}.{}".format(MODEL_MODULE, MODEL_CLASS)
    s += "Torch from: {}".format(torch.__file__)
    s += "MCTS from: {}".format(cmcts.__file__)
    s += "MCTS number of threads: {}".format(THREADS)
    s += "Number of games: {}".format(EVAL_GAMES)
    s += "Timeout: {}m".format(EVAL_GAMES)
    return s

