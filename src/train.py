import sys
sys.path.append('..')
from config import *
from importlib import import_module
model_module = import_module(MODEL_MODULE)
import cmcts
import pdb
import os
import glob
import torch
import logging
import numpy as np
from datetime import datetime
import tools

def train(model_class, param_file, new_param_file, data_files):
    logging.basicConfig(format='%(levelname)s: %(message)s',
            filename="{}/train_{}.log".format(LOG_PATH, os.path.basename(new_param_file).replace(".pyt",'')),
            level=logging.DEBUG)
    logging.info("########################################")

    logging.info("Using model {} for training".format(param_file))
    logging.info("New model is {}".format(new_param_file))

    model = model_class()

    if not os.path.isfile(param_file):
        logging.error("File \"{}\" does not exists".format(param_file))
        return False

    params = torch.load(param_file)
    model.load_state_dict(params['state_dict'])

    # get window
    logging.info("Training files:")
    if len(data_files) == 0:
        logging.error("No data files provided")
        return False

    for d in data_files:
        logging.info("\t{}".format(d))

    data_list = [ tools.get_rotations(tools.get_unique(np.load(d))) for d in data_files ]
    #data_list = [ tools.get_rotations(tools.get_unique(np.load(d))) for d in data_files ]

    data = np.concatenate(data_list)

    del data_list

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    criterion_pi = torch.nn.BCEWithLogitsLoss()
    criterion_v  = torch.nn.MSELoss()

    # show progress 100 times per epoch
    view_step = data.size//BATCH_SIZE//10
    logging.info("Data size {} divided into {} batches of size {}".format(data.size, data.size//BATCH_SIZE, BATCH_SIZE))

    for e in range(EPOCHS):
        np.random.shuffle(data)
        logging.info("Running epoch {}/{}".format(e,EPOCHS))
        acc_vloss = 0
        acc_ploss = 0
        acc_loss  = 0
        for i in range(data.size//BATCH_SIZE):
            #batch_ids = np.random.choice(data.size, BATCH_SIZE)
            batch_input   = torch.from_numpy(data[i*BATCH_SIZE:(1+i)*BATCH_SIZE]['board'])
            batch_vlabels = torch.from_numpy(data[i*BATCH_SIZE:(1+i)*BATCH_SIZE]['r']).reshape(-1,1)
            batch_plabels = torch.from_numpy(data[i*BATCH_SIZE:(1+i)*BATCH_SIZE]['pi'])
            #pdb.set_trace()

            v, pi = model(batch_input)

            vloss = criterion_v(v, batch_vlabels)
            ploss = criterion_pi(pi, batch_plabels)
            loss = vloss + ploss

            acc_vloss += vloss
            acc_ploss += ploss
            acc_loss  += loss

            if i % view_step == view_step -1:
                logging.info("Epoch: {}\nIteration: {}\nvalue accuracy: {}\npi accuracy:{}\nTotal loss:{}".
                        format(e,i,acc_vloss/i,acc_ploss/i,acc_loss/i))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    logging.info("Saving model to {}".format(new_param_file))
    torch.save({
        'state_dict' : model.state_dict(),
        },new_param_file)
    logging.info("####################END#################")
    return True

if __name__ == "__main__":
    param_file, new_param_file, data_files = tools.info_train()
    model_class = getattr(model_module, MODEL_CLASS)
    if train(model_class, param_file, new_param_file, data_files):
        print("New model parameters were saved to {}".format(new_param_file))
    else:
        print("Training failed, check for errors {}/train_{}.log".format(LOG_PATH, new_param_file))
