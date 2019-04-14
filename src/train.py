import sys
sys.path.append('..')
from config import *
from importlib import import_module
model_module = import_module(MODEL_MODULE)
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
    logging.info(tools.train_config())
    tools.config_save(os.path.basename(new_param_file).replace(".pyt", ''))
    print(tools.train_config())

    cuda = torch.cuda.is_available()
    logging.info("Is cuda avalable {}".format(cuda))
    logging.info("Using model {} for training".format(param_file))
    logging.info("New model is {}".format(new_param_file))

    model = model_class()
    model.train()

    if param_file is None or not os.path.isfile(param_file):
        # if no parameters found continue with randomly initialised parameters
        logging.warning("File \"{}\" does not exists".format(param_file))
    else:
        print(param_file)
        try:
            params = torch.load(param_file)
            if cuda:
                model.load_state_dict(params['state_dict']).cuda()
                logging.info("GPU {}",torch.cuda.get_device_name())
            else:
                model.load_state_dict(params['state_dict'])
        except RuntimeError as e:
            # probably architecture has changed
            logging.warning("Could not load parametrs")
            logging.warning(e)

    # get window
    logging.info("Training files:")
    if len(data_files) == 0:
        logging.error("No data files provided")
        return False

    for d in data_files:
        logging.info("\t{}".format(d))

    # loading data
    data_list = [ np.load(d) for d in data_files ]

    data = np.concatenate(data_list)
    logging.info("Original data size {}".format(data.size))
    # get all possible board rotations and remove duplicate board positions
    data = tools.get_unique(tools.get_rotations(data))
    logging.info("Augmented data size {}".format(data.size))
    #data = tools.get_unique(data)
    del data_list

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    criterion_pi = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_v  = torch.nn.MSELoss()

    # step in which log will be created
    iterations = data.size//BATCH_SIZE
    if iterations < 25:
        view_step = 1
    else:
        view_step = iterations//20 # progress will be displayed 20times per epoch

    logging.info("Iterations per epoch {}".format(iterations))
    logging.info("Batch size {}".format(BATCH_SIZE))
    start_train = datetime.now()
    logging.info("Starting at {}".format(start_train))

    for e in range(EPOCHS):
        acc_vloss = 0
        acc_ploss = 0
        acc_loss  = 0
        it = 0
        start_epoch = datetime.now()
        logging.info("Running epoch {}/{}".format(e+1,EPOCHS))
        acc_batchtime = None
        for i in range(iterations):
            batch_ids = np.random.choice(data.size, BATCH_SIZE)
            batch_input   = torch.from_numpy(data[batch_ids]['board'])
            batch_vlabels = torch.from_numpy(data[batch_ids]['r']).reshape(-1, 1, 1, 1)
            batch_plabels = torch.from_numpy(data[batch_ids]['pi'])

            if cuda:
                batch_input.cuda()
                batch_vlabels.cuda()
                batch_plabels.cuda()
            v, p = model(batch_input)

            # computing loss
            vloss = criterion_v(v, batch_vlabels)
            ploss = criterion_pi(p, (batch_plabels+1e-15))
            loss = vloss + ploss

            it += 1
            acc_vloss += vloss.item()
            acc_ploss += ploss.item()
            acc_loss  += loss.item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % view_step == view_step -1:
                logging.info("E:{}/{} I:{}/{} AccVALUE loss: {}".format(e+1, EPOCHS, i+1, iterations, acc_vloss))
                logging.info("E:{}/{} I:{}/{} AccPI    loss: {}".format(e+1, EPOCHS, i+1, iterations, acc_ploss))
                logging.info("E:{}/{} I:{}/{} AccTOTAL loss: {}".format(e+1, EPOCHS, i+1, iterations, acc_ploss))

        # end of epoch
        end = datetime.now()
        logging.info("Epoch {} took {}".format(e, end-start_epoch))
        logging.info("Epoch:{} AccValue loss: {}".format(e, acc_vloss/it))
        logging.info("Epoch:{} AccPi loss:{}".format(e, acc_vloss/it))
        logging.info("Epoch:{} Total loss:{}".format(e, acc_vloss/it))
        logging.info("Epoch:{} checkpoint".format(new_param_file))
        torch.save({
            'state_dict' : model.state_dict(),
            },new_param_file)

    # saving new parameters
    logging.info("Training took {}".format(end-start_train))
    logging.info("Saving model to {}".format(new_param_file))
    torch.save({
        'state_dict' : model.state_dict(),
        },new_param_file)
    logging.info("####################END#################")
    return True

if __name__ == "__main__":
    param_file = tools.get_params()
    new_param_file = tools.get_new_params()
    data_files = tools.get_data()
    model_class = getattr(model_module, MODEL_CLASS)
    if train(model_class, param_file, new_param_file, data_files):
        print("New model parameters were saved to {}".format(new_param_file))
    else:
        print("Training failed, check for errors {}/train_{}.log".format(LOG_PATH, new_param_file))
    if not tools.dry_run():
        tools.init_generation()
