import sys
sys.path.append('..')
from config import SHAPE, SIZE, CMCTS_SITE_PATH, TRAIN_SAMPLES
import site
site.addsitedir(CMCTS_SITE_PATH)
from time import time
import numpy as np
import json
info_file = "../.info"

def rand_uint32():
    return np.random.randint(np.iinfo(np.int32).max, dtype=np.uint32).item()

def info_read():
    # info will be a dictionary containig iteration, generation and best
    # generation is list of generation
    try:
        with open(info_file, "r") as fp:
            info = json.load(fp)
            return info['iteration'], info['generation'], info['best']
    except (IOError, ValueError):
        with open(info_file, "w") as fp:
            info = {
                    'iteration' : -1,
                    'generation' : -1,
                    'best' : -1
                    }
            json.dump(info, fp)
            return info['iteration'], info['generation'], info['best']

def info_iteration():
    with open(info_file, "r+") as fp:
        try:
            info = json.load(fp)
        except ValueError:
            # creating info file
            info = {
                    'iteration' : -1,
                    'generation' : -1,
                    'best' : -1
                    }
        fp.seek(0)
        fp.truncate()
        info['iteration'] = info['iteration'] + 1
        json.dump(info, fp)

        return info['iteration']

def info_generation():
    with open(info_file, "r+") as fp:
        info = json.load(fp)
        info['generation'] = info['generation'] + 1
        fp.seek(0)
        fp.truncate()
        json.dump(info, fp)

        return info['generation']

def info_best(best):
    with open(info_file, "r+") as fp:
        info = json.load(fp)
        info['best'] = best
        fp.seek(0)
        fp.truncate()
        json.dump(info, fp)

        return info['best']

def info_set(iteration=None, generation=None, best=None):
    with open(info_file, "r+") as fp:
        try:
            info = json.load(fp)
        except ValueError:
            # creating info file
            info = {
                    'iteration' : -1,
                    'generation' : -1,
                    'best' : -1
                    }
        if iteration is not None:
            info['iteration'] = iteration
        if generation is not None:
            info['generation'] = generation
        if best is not None:
            info['best'] = best

        fp.seek(0)
        fp.truncate()
        json.dump(info, fp)

