import sys
sys.path.append('..')
from config import *
import numpy as np
import json
import tarfile
import pdb
from glob import glob
import os
from datetime import datetime
import torch
import shutil

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
    return
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
    """ averages duplicate board positions
    """
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
    """ creates rotations of board
    """
    rotations = []
    dt = data.dtype
    shape = data['board'].shape[-1]

    for d in data:
        board = d['board']
        pi = d['pi'].reshape(shape,shape)

        for r in range(7):
            if r == 3:
                board = np.flip(board, 1)
                pi    = np.flip(pi, 0)
            else:
                board = np.rot90(board, axes=(1,2))
                pi    = np.rot90(pi)

            rotations.append(np.array((board, pi.reshape(-1), d['r']), dtype=dt))

    return np.concatenate([data, np.stack(rotations)])

def get_new_params():
    """ returns file name where new parameters will be saved
    """
    params = "{}/{}{}_{}.pyt".format(
                PARAM_PATH, MODEL_CLASS, SHAPE, datetime.now().strftime("%m%d_%H-%M-%S"))
    return params

def get_params():
    """ returns parameters to best model
    """
    if 'PARAM' in os.environ:
        params = get_param_file(os.environ['PARAM'])
    else:
        params = get_best()
        if params is None:
            print("No parameters found")
    return params

def get_versus():
    """ returns parameters against which will be best model playing
    """
    if 'VERSUS' in os.environ:
        params = get_param_file(os.environ['VERSUS'])
    else:
        params = get_latest()
        if params is None:
            print("No parameters found")
    return params

def get_data():
    """ returns list of filenames containing data for training
        also removes obsolete data
    """
    if 'DATA' in os.environ:
        data = os.environ['DATA'].split(',').strip()
    else:
        data = glob("{}/{}{}_*[!.tgz]".format(DATA_PATH, MODEL_CLASS, SHAPE))
        data.sort()
        if len(data) > 0:
            generation = int(data[-1][-3:])
        else:
            raise RuntimeError("No data for training found")

        if len(data) > 4:
            window = min(WINDOW[0]+(generation-3)//WINDOW[2], WINDOW[1])
            print("Window size is {}".format(window))
            print("Number of data dirs is {}".format(len(data)))
            if len(data) - window > 0:
                for d in data[0:-window]:
                    # make gzip and remove
                    print("Removing file {}".format(d))
                    rm(d)
            data = data[-window:]

    data_files = []
    for d in data:
        if not os.path.exists(d):
            print("File or directory {} does not exist".format(d))
            print("Excluding it from data list")
            continue
        if "tgz" in d:
            continue
        if os.path.isdir(d):
            data_files.extend(glob("{}/*.npy".format(os.path.realpath(d))))
        else:
            data_files.append(os.path.realpath(d))

    return data_files

def rm(files):
    if type(files) is list:
        if len(files) < 1:
            raise RuntimeError("RM invalid argument")
        name = os.path.basename(files[0])
    else:
        name = os.path.basename(files)
        files = [files]
    with tarfile.open("{}/{}.tgz".format(DATA_PATH, name), "w:gz") as tar:
        for f in files:
            tar.add(f, recursive=True)
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
            else:
                print("Could not remove {}. Not dir or file".format(f))


def init_generation():
    """ creats a directory for new generation of data
    """
    data_dirs = glob("{}/{}{}_*/".format(DATA_PATH, MODEL_CLASS, SHAPE))
    data_dirs.sort()
    if len(data_dirs) == 0:
        generation = -1
    else:
        generation = int(data_dirs[-1].replace('/','')[-3:])
    print("Last generation was: {}".format(generation))
    data_dir = "{}/{}{}_{}_gen{:03}".format(
            DATA_PATH,
            MODEL_CLASS,
            SHAPE,
            datetime.now().strftime("%m%d_%H-%M"),
            generation+1)
    print("New this generation will be in {}".format(data_dir))
    try:
        os.mkdir(data_dir)
    except OSError as e:
        print("Creating directory failed")
        print(e)
        sys.exit(1)

    if generation != -1:
        # if it is the first generation there will be no parameter file saved yet
        with open("{}/parameters.txt".format(data_dir), "w+") as f:
            f.write(get_best())

    return data_dir

def get_new_data():
    """ returns name of new data file
    """
    # if self-play is runnig in multiple processes
    # setting sequence number will avoid name colisions
    if 'SEQUENCE' in os.environ:
        seq = os.environ['SEQUENCE']
    else:
        seq = 0

    # finding directory where will be data saved
    data_dir = glob("{}/{}{}_*/".format(DATA_PATH, MODEL_CLASS, SHAPE))
    data_dir.sort()
    if len(data_dir) == 0:
        data_dir = init_generation()
    else:
        data_dir = data_dir[-1]

    file_name = "{}/{}s{}".format(data_dir, datetime.now().strftime("%m%d_%H-%M-%S"), seq)

    return file_name

def dry_run():
    """ if dry run is set best parameters won't be changed
        returns True if dry run is set
                False if not
    """
    if 'DRY_RUN' in os.environ:
        return True
    return False

def get_latest():
    """ returns latest parameters
    """
    param_files = glob("{}/{}{}_*.pyt".format(
        PARAM_PATH, MODEL_CLASS, SHAPE))
    if len(param_files) == 0:
        return None

    if not os.path.isabs(param_files[-1]):
        return "{}/{}".format(PARAM_PATH, param_files[-1])
    return param_files[-1]

def get_param_file(name):
    """ function will check if parameters exists
        returns absolute path to parameters
    """
    if os.path.isfile(name):
        return name
    abs_name = "{}/{}".format(PARAM_PATH, os.path.basename(name))
    if os.path.isfile(abs_name):
        return abs_name
    else:
        raise RuntimeError("File {} or {} does not exists".format(name, abs_name))

def get_best():
    """ returns best parameters
    """
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
            path = os.path.basename(content['path'])
            path = os.path.join(PARAM_PATH, path)
            if not os.path.isfile(path):
                print("Could not found parameters")
                print("Running with default parameters")
            return path

        return content['path']

def set_best(path):
    """ sets new best parameters
    """
    if dry_run():
        print("Dry run: setting {} as best".format(path))
        return
    with open(PARAM_BEST, "w") as fp:
        if not os.path.isabs(path):
            path = "{}/{}".format(PARAM_PATH, path)
        if not os.path.isfile(path):
            raise RuntimeError("File {} does not exst".format(path))

        path = os.path.normpath(path)
        content = { "path" : path }
        json.dump(content, fp)

def init_param_file():
    """ initialises file containing path to file with best parameters
    """
    open(PARAM_BEST, 'a').close()

def self_play_config():
    s = ""
    s += "Number of training samples to generate: {}\n".format(TRAIN_SAMPLES)
    s += "Timeout for self play: {} minutes\n".format(TIMEOUT_SELF_PLAY)
    s += "Data destination: {}\n".format(DATA_PATH)
    s += "Threads: {}\n".format(DATA_PATH)
    s += "Parameters loaded from: {}\n".format(PARAM_PATH)
    s += "Model from: {}.{}\n".format(MODEL_MODULE, MODEL_CLASS)
    s += "Torch from: {}\n".format(torch.__file__)
    s += "MCTS number of threads: {}\n".format(THREADS)
    return s

def train_config():
    s = ""
    s += "Data source: {}\n".format(DATA_PATH)
    s += "Parameters loaded from: {}\n".format(PARAM_PATH)
    s += "Model from: {}.{}\n".format(MODEL_MODULE, MODEL_CLASS)
    s += "Torch from: {}\n".format(torch.__file__)
    s += "Learning rate: {}\n".format(LR)
    s += "Epochs: {}\n".format(EPOCHS)
    s += "Batch size: {}\n".format(BATCH_SIZE)
    s += "Starting window size: {}\n".format(WINDOW[0])
    s += "Max window size: {}\n".format(WINDOW[1])
    s += "Window size incremented every {} generation\n".format(WINDOW[2])
    return s

def eval_config():
    s = ""
    s += "Parameters loaded from: {}\n".format(PARAM_PATH)
    s += "Model from: {}.{}\n".format(MODEL_MODULE, MODEL_CLASS)
    s += "Torch from: {}\n".format(torch.__file__)
    s += "MCTS number of threads: {}\n".format(THREADS)
    s += "Number of games: {}\n".format(EVAL_GAMES)
    s += "Timeout: {} minutes\n".format(EVAL_GAMES)
    return s

def config_load(conf_file=LOAD_CONFIG):
    print(CONFIG_PATH)
    print(os.path.basename(conf_file))
    print(conf_file)
    conf_file = "{}/{}".format(CONFIG_PATH, os.path.basename(conf_file))
    if not os.path.isfile(conf_file):
        raise RuntimeError("Configuration file {} not found".format(conf_file))
    print("Loading {}".format(conf_file))
    shutil.move(conf_file, "{}/congig.py".format(ROOT))

def config_save(conf_name):
    shutil.copyfile('config.py'.format(ROOT), "{}/config/conf_{}.py".format(ROOT, conf_name))
