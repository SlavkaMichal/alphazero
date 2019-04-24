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

    cuda = torch.cuda.is_available()
    logging.info("Is cuda avalable {}".format(cuda))
    logging.info("Using model {} for training".format(param_file))
    logging.info("New model is {}".format(new_param_file))

    config = tools.get_param_conf()
    model_module = import_module(config.MODEL_MODULE)
    model_class = getattr(model_module, config.MODEL_CLASS)
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
            if 'PARAM' in os.environ and LOAD_CONFIG_PARAM:
                logging.error("Supplied parameters does not match with configuration file")
                sys.exit(1)


    # get window
    logging.info("Training files:")
    if len(data_files) == 0:
        logging.error("No data files provided")
        return False

    for d in data_files:
        logging.info("\t{}".format(d))

    # loading data
    data_list = []
    for d in data_files:
        try:
            data_list.append(np.load(d))
        except ValueError as e:
            logging.error("Could not load file {}".format(d))
            logging.error(e)

    data = np.concatenate(data_list)
    logging.info("Original data size {}".format(data.size))
    # get all possible board rotations and remove duplicate board positions
    if config.ROTATIONS:
        logging.info("Augmenting data")
        logging.info("Original data size {}".format(data.size))
        data = tools.get_rotations(data)
        logging.info("Augmented data size {}".format(data.size))
    data = tools.get_unique(data)
    logging.info("Data size {}".format(data.size))
    np.random.shuffle(data)
    validation_size = int(data.size*0.05)
    validation = data[:validation_size]
    data = data[validation_size:]
    logging.info("Training data size {}".format(data.size))
    logging.info("Validation data size {}".format(validation.size))
    #data = tools.get_unique(data)
    del data_list

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    criterion_pi = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_v  = torch.nn.MSELoss()

    # step in which log will be created
    iter_train = data.size//config.BATCH_SIZE
    iter_train = iter_train if iter_train > 0 else 1
    if iter_train < VIEW_STEP:
        view_step = 1
    else:
        view_step = iter_train//VIEW_STEP # progress will be displayed 20times per epoch

    iter_valid = validation.size//config.BATCH_SIZE
    iter_valid = iter_valid if iter_valid > 0 else 1
    if iter_valid < VIEW_STEP:
        view_step_valid = 1
    else:
        view_step_valid = iter_valid//VIEW_STEP


    logging.info("Iterations per epoch {}".format(iter_train))
    logging.info("Batch size {}".format(config.BATCH_SIZE))
    start_train = datetime.now()
    logging.info("Starting at {}".format(start_train))

    for e in range(config.EPOCHS):
        acc_vloss = 0
        acc_ploss = 0
        acc_loss  = 0
        it = 0
        start_epoch = datetime.now()
        logging.info("Running epoch {}/{}".format(e+1,config.EPOCHS))
        acc_batchtime = None
        model.train()
        for i in range(iter_train):
            batch_ids = np.random.choice(data.size, config.BATCH_SIZE)
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
                progress = int(round(it/iter_train*100))
                logging.info("E:{}/{} I:{:3}% AccVALUE loss: {}".format(e+1, config.EPOCHS, progress, acc_vloss/it))
                logging.info("E:{}/{} I:{:3}% AccPI    loss: {}".format(e+1, config.EPOCHS, progress, acc_ploss/it))
                logging.info("E:{}/{} I:{:3}% AccTOTAL loss: {}".format(e+1, config.EPOCHS, progress, acc_loss/it))

        logging.info("Epoch:{} AccPi loss:{}".format(e, acc_ploss/it))
        logging.info("Epoch:{} AccValue loss: {}".format(e, acc_vloss/it))
        logging.info("Epoch:{} Total loss:{}".format(e, acc_loss/it))

        # validation
        acc_vloss = 0
        acc_ploss = 0
        acc_loss  = 0
        it = 0
        start_valid = datetime.now()
        logging.info("Running validation of epoch {}/{}".format(e+1,config.EPOCHS))
        acc_batchtime = None
        model.eval()
        for i in range(iter_valid):
            batch_ids = np.random.choice(validation.size, config.BATCH_SIZE)
            batch_input   = torch.from_numpy(validation[batch_ids]['board'])
            batch_vlabels = torch.from_numpy(validation[batch_ids]['r']).reshape(-1, 1, 1, 1)
            batch_plabels = torch.from_numpy(validation[batch_ids]['pi'])

            if cuda:
                batch_input.cuda()
                batch_vlabels.cuda()
                batch_plabels.cuda()
            with torch.no_grad():
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
            if i % view_step_valid == view_step_valid -1:
                progress = int(round(it/iter_valid*100))
                logging.info("E:{}/{} I:{:3}% ValidAccVALUE loss: {}".format(e+1, config.EPOCHS, progress, acc_vloss/it))
                logging.info("E:{}/{} I:{:3}% ValidAccPI    loss: {}".format(e+1, config.EPOCHS, progress, acc_ploss/it))
                logging.info("E:{}/{} I:{:3}% ValidAccTOTAL loss: {}".format(e+1, config.EPOCHS, progress, acc_loss/it))

        # end of epoch
        end = datetime.now()
        logging.info("Epoch {} took {}".format(e, end-start_epoch))
        logging.info("Epoch:{} ValidAccPi    loss:{}".format(e, acc_ploss/it))
        logging.info("Epoch:{} ValidAccValue loss: {}".format(e, acc_vloss/it))
        logging.info("Epoch:{} ValidTotal    loss:{}".format(e, acc_loss/it))
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
    if train(param_file, new_param_file, data_files):
        print("New model parameters were saved to {}".format(new_param_file))
    else:
        print("Training failed, check for errors {}/train_{}.log".format(LOG_PATH, new_param_file))
