#################################################
# this file is only used for debugging purposes #
#################################################

import cmcts
import torch
import numpy as np
import sys
sys.path.append('..')
from general_config import *
import torch
import os
from torch import nn
from importlib import import_module
from tools import rand_uint32
import tools
from datetime import datetime

print(tools.__file__)
conf = tools.get_param_conf()
model_module = import_module(conf.MODEL_MODULE)
model_class = getattr(model_module, conf.MODEL_CLASS)
hmcts = cmcts.mcts(cpuct=conf.CPUCT)
mcts = cmcts.mcts(cpuct=conf.CPUCT)
model = model_class(conf)
example = torch.rand(1,2,SHAPE,SHAPE)
param_file = tools.get_params()
params = torch.load(param_file)
model.load_state_dict(params['state_dict'])
with torch.no_grad():
    traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("tmp.pt")
mcts.set_alpha_default()
hmcts.set_alpha_default()
mcts.set_params("tmp.pt")

def run(mcts, sims):
    mcts.print_node([])
    mcts.simulate(sims)
    mcts.print_node([])
    move = np.argmax(mcts.get_prob())
    mcts.make_move(move)
    print(mcts)
    mcts.print_node([])
#print("set here")
#mcts.set_predictor(model_wraper, model)
#print("sim here")

#mcts.simulate(1000)
