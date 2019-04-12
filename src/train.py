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


    params = torch.load(param_file)

    if not os.path.isfile(param_file):
        logging.warning("File \"{}\" does not exists".format(param_file))
    else:
        try:
            if cuda:
                model.load_state_dict(params['state_dict']).cuda()
                logging.info("GPU {}",torch.cuda.get_device_name())
            else:
                model.load_state_dict(params['state_dict'])
        except RuntimeError as e:
                logging.warning("Could not load parametrs")
                logging.warning(e)

    # get window
    logging.info("Training files:")
    if len(data_files) == 0:
        logging.error("No data files provided")
        return False

    for d in data_files:
        logging.info("\t{}".format(d))

    data_list = [ np.load(d) for d in data_files ]

    data = np.concatenate(data_list)
    #data = tools.get_rotations(tools.get_unique(data))
    data = tools.get_unique(data)
    del data_list

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    criterion_pi = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_v  = torch.nn.MSELoss()

    view_step = data.size//20
    logging.info("Data size {} divided into {} batches of size {}".
            format(data.size, data.size//BATCH_SIZE, BATCH_SIZE))
    start_train = datetime.now()
    logging.info("Starting at {}".format(start_train))

    for e in range(EPOCHS):
        acc_vloss = 0
        acc_ploss = 0
        acc_loss  = 0
        it = 0
        start_epoch = datetime.now()
        #np.random.shuffle(data)
        logging.info("Running epoch {}/{}".format(e,EPOCHS))
        acc_batchtime = None
        for i in range(data.size):
            #batch_ids = np.random.choice(data.size, BATCH_SIZE)
            #batch_input   = torch.from_numpy(data[batch_ids]['board'])
            #batch_vlabels = torch.from_numpy(data[batch_ids]['r']).reshape(-1)
            #batch_plabels = torch.from_numpy(data[batch_ids]['pi'])

            # batch size is one
            batch_input   = torch.from_numpy(data[i:i+1]['board'])
            batch_vlabels = torch.from_numpy(data[i:i+1]['r']).reshape(-1)
            batch_plabels = torch.from_numpy(data[i:i+1]['pi'])
            if cuda:
                batch_input.cuda()
                batch_vlabels.cuda()
                batch_plabels.cuda()
            v, p = model(batch_input)

            # computing loss
            vloss = criterion_v(v, batch_vlabels)
            ploss = criterion_pi(p, (batch_plabels+1e-15))
            loss = vloss + ploss
            #ploss = -((pi+1e-15).log()*batch_plabels).sum()/pi.shape[0]

            it += 1
            acc_vloss += vloss.item()
            acc_ploss += ploss.item()
            acc_loss  += loss.item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % view_step == view_step -1:
                logging.info("step:\n\tVALUE loss:\t{}\n\tPI loss:\t{}\n\tTOTAL loss:\t{}".
                        format( vloss.item(),ploss.item(), loss.item()))
                logging.info("diffV {}".format(v[0].item() - batch_vlabels[0].item()))
                logging.info("diffP {}".format((p[0].exp() - batch_plabels[0]).sum().item()))

        # end of epoch
        end = datetime.now()
        logging.info("Epoch {} took {}".format(e, end-start_epoch))
        logging.info("AccValue loss: {}\nAccPi loss:{}\nTotal loss:{}".
                        format(acc_vloss/it,acc_ploss/it,acc_loss/it))

    # saving new parameters
    logging.info("Training took {}".format(end-start_train))
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
