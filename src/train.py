import sys
sys.path.append('..')
from config import *
import site
site.addsitedir(LOCAL_SITE_PATH)
from model import simplerNN
import cmcts
import os
import glob
import torch
import numpy as np
from datetime import datetime
import tools

def train(model_class):
    model_name = "{}{}".format(model_class.__name__,SHAPE)
    # read info
    iteration, last_generation, train_generation = tools.info_read()

    logging.basicConfig(format='%(levelname)s: %(message)s', filename="../logs/train{:03}.log".format(new_generation), level=logging.DEBUG)
    logging.info("Using generation {} for training".format(train_generation))
    logging.info("Next generation is {}".format(last_generation+1))

    # model name:
    #   [class name of model][board size]_[generation]
    if last_generation == -1:
        logging.info("No model parameters to train")
        return
    if train_generation == -1:
        logging.info("Using last generation for training")
        tools.info_best(last_generation)
        train_generation = last_generation


    # parameters name:
    #   [class name of model][board size]_[generation].pt

    model = model_class()

    param_file = "../model/{}_{:03}.pt".format(model_name, train_generation)
    logging.info("Loading model parameters from {}".format(param_file))
    params = torch.load(param_file)
    model.load_state_dict(params['state_dict'])
    #filename = "%s%d_%03d"%(model_class.__name__, SIZE, iteration)

    files = glob.glob("../data/{}_*".format(model_name))
    if len(files) == 0:
        logging.info("No training files in data folder")
        logging.info("Number of iterations expected: {}".format(iteration+1))
        return

    # get window
    files.sort()
    size = min(WINDOW[0]+last_generation//WINDOW[1],WINDOW[3])
    logging.info("Window size: {}".format(size))
    files = files[-size:]

    data_list = [ np.load(f) for f in files ]

    data = np.concatenate(data_list)
    np.random.shuffle(data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#    batch_input = np.empty((BATCH_SIZE, 2, SIZE, SIZE), dtype='f4')
    criterion_pi = torch.nn.BCEWithLogitsLoss()
    criterion_v  = torch.nn.MSELoss()

    #show progress 100 times per epoch
    view_step = data.size//BATCH_SIZE

    for e in range(EPOCHS):
        acc_vloss = 0
        acc_ploss = 0
        for i in range(data.size//BATCH_SIZE):
            #batch_ids = np.random.choice(data.size, BATCH_SIZE)
            batch_input = torch.from_numpy(data[i:i*BATCH_SIZE]['board'])
            batch_vlabels = torch.from_numpy(data[i:i*BATCH_SIZE]['r']).reshape(-1,1)
            batch_plabels = torch.from_numpy(data[i:i*BATCH_SIZE]['pi'])

            v, pi = model(batch_input)

            vloss = criterion_v(v, batch_vlabels)
            ploss = criterion_pi(pi, batch_plabels)
            loss = vloss + ploss

            acc_vloss += vloss
            acc_ploss += ploss

             show progress
            if i % view_step == view_step -1:
                print("Epoch: {}\nIteration: {}\nvalue accuracy: {}\npi accuracy:{}".
                        format(e,i,acc_vloss/i,acc_ploss/i))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info("Epoch: {}\nIteration: {}\nvalue accuracy: {}\npi accuracy:{}".
           format(e,i,acc_vloss/i,acc_ploss/i))

    new_generation = tools.info_generation()
    torch.save({
        'state_dict' : model.state_dict(),
        'iteration'  : iteration
        },"{}_{:03d}.pt".format(model_name,new_generation))

if __name__ == "__main__":
    train(small_nn)
