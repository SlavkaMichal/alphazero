import sys
sys.path.append('..')
from general_config import *
from importlib import import_module
import pdb
import os
import glob
import torch
import logging
import numpy as np
from datetime import datetime
import tools

def train(param_file, new_param_file, data_files):
    logging.basicConfig(format='%(levelname)s: %(message)s',
            filename="{}/train_{}.log".format(LOG_PATH, os.path.basename(new_param_file).replace(".pyt",'')),
            level=logging.DEBUG)
    logging.info("########################################")
    logging.info(tools.str_conf())
    tools.config_save(os.path.basename(new_param_file).replace(".pyt", ''))
    print(tools.str_conf())

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Device: {}".format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device('cpu')
        logging.info("Device: CPU")

    logging.info("Using model {} for training".format(param_file))
    logging.info("New model is {}".format(new_param_file))

    config = tools.get_param_conf()
    model_module = import_module(config.MODEL_MODULE)
    model_class = getattr(model_module, config.MODEL_CLASS)
    model = model_class(config)
    model.train()

    if param_file is None or not os.path.isfile(param_file):
        # if no parameters found continue with randomly initialised parameters
        logging.warning("File \"{}\" does not exists".format(param_file))
    else:
        print(param_file)
        try:
            params = torch.load(param_file)
            model.load_state_dict(params['state_dict'])
        except RuntimeError as e:
            # probably architecture has changed
            logging.warning("Could not load parametrs")
            logging.warning(e)
            if 'PARAM' in os.environ and LOAD_CONFIG_PARAM:
                logging.error("Supplied parameters does not match with configuration file")
                sys.exit(1)

    model.to(device=device)

    # check data files
    logging.info("Training files:")
    if len(data_files) == 0:
        logging.error("No data files provided")
        return False

    for i, c in enumerate(data_files):
        logging.info("\tChunk: {}".format(i))
        for d in c:
            logging.info("\t{}".format(d))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    criterion_pi = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_v  = torch.nn.MSELoss()

    logging.info("Batch size {}".format(config.BATCH_SIZE))
    start_train = datetime.now()
    logging.info("Starting at {}".format(start_train))

    for e in range(config.EPOCHS):
        model.train()
        acc_vloss_epoch = 0
        acc_ploss_epoch = 0
        acc_loss_epoch  = 0
        iter_total = 0

        start_epoch = datetime.now()
        logging.info("Running epoch {}/{}".format(e+1,config.EPOCHS))
        logging.info("Data chunk {}/{}".format(e+1,config.EPOCHS))

        for data_idx, chunk in enumerate(data_files):
            start_data_file = datetime.now()

            data = tools.load_data(chunk)

            logging.info("Data size: {}".format(data.size))
            if data.size == 0:
                logging.warning("No data in chunk")
                continue

            # get all possible board rotations and remove duplicate board positions
            if config.ROTATIONS:
                logging.info("Augmenting data")
                data = tools.get_rotations(data)
                logging.info("Augmented data size: {}".format(data.size))
            data = tools.get_unique(data)
            logging.info("Unique data size: {}".format(data.size))

            iter_train = data.size//config.BATCH_SIZE
            iter_train = iter_train if iter_train > 0 else 1

            acc_vloss = 0
            acc_ploss = 0
            acc_loss  = 0

            for i in range(iter_train):
                batch_ids = np.random.choice(data.size, config.BATCH_SIZE)
                batch_input   = torch.from_numpy(data[batch_ids]['board']).to(device)
                batch_vlabels = torch.from_numpy(data[batch_ids]['r']).reshape(-1, 1, 1, 1).to(device)
                batch_plabels = torch.from_numpy(data[batch_ids]['pi']).to(device)

                v, p = model(batch_input)

                # computing loss
                vloss = criterion_v(v, batch_vlabels)
                ploss = criterion_pi(p, (batch_plabels+1e-15))
                loss = vloss + ploss

                acc_vloss += vloss.item()
                acc_ploss += ploss.item()
                acc_loss  += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc_vloss_epoch += acc_vloss
            acc_ploss_epoch += acc_ploss
            acc_loss_epoch  += acc_loss
            iter_total      += iter_train

            end = datetime.now()
            logging.info("E:{}/{} D:{}/{} took: {}".format(e+1, config.EPOCHS, data_idx, len(data_files), end-start_data_file))

            logging.info("E:{}/{} D:{}/{} DataAccValue loss: {}".format(e+1, config.EPOCHS, data_idx, len(data_files), acc_vloss/iter_train))
            logging.info("E:{}/{} D:{}/{} DataAccPi    loss: {}".format(e+1, config.EPOCHS, data_idx, len(data_files), acc_ploss/iter_train))
            logging.info("E:{}/{} D:{}/{} DataAccTotal loss: {}".format(e+1, config.EPOCHS, data_idx, len(data_files), acc_loss/iter_train))

            logging.info("E:{}/{} D:{}/{} EpochAccValue loss: {}".format(e+1, config.EPOCHS, data_idx, len(data_files), acc_vloss_epoch/iter_total))
            logging.info("E:{}/{} D:{}/{} EpochAccPi    loss: {}".format(e+1, config.EPOCHS, data_idx, len(data_files), acc_ploss_epoch/iter_total))
            logging.info("E:{}/{} D:{}/{} EpochAccTotal loss: {}".format(e+1, config.EPOCHS, data_idx, len(data_files), acc_loss_epoch/iter_total))

        logging.info("Epoch: {}/{} took {}".format(e+1, config.EPOCHS, end-start_epoch))
        logging.info("E:{}/{} EpochAccVALUE loss: {}".format(e+1, config.EPOCHS, acc_vloss_epoch/iter_total))
        logging.info("E:{}/{} EpochAccPI    loss: {}".format(e+1, config.EPOCHS, acc_ploss_epoch/iter_total))
        logging.info("E:{}/{} EpochAccTOTAL loss: {}".format(e+1, config.EPOCHS, acc_loss_epoch/iter_total))
        logging.info("E:{}/{} D:{}/{} checkpoint: {}".format(e+1, config.EPOCHS, data_idx, len(data_files), "{}.tmp{}".format(new_param_file, e)))
        torch.save({
            'state_dict' : model.state_dict(),
            },"{}.tmp{}".format(new_param_file, e))

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
    if train(param_file, new_param_file, data_files):
        print("New model parameters were saved to {}".format(new_param_file))
    else:
        print("Training failed, check for errors {}/train_{}.log".format(LOG_PATH, new_param_file))
