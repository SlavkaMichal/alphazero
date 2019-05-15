#!/usr/bin/python
import sys
import os
sys.path.append('..')
import site
from general_config import *
import numpy as np
import cmcts
import argparse
import server as cmds
import tools
import torch
from tools import rand_uint32
from server import Server
from importlib import import_module

HOST = '127.0.0.1'
PORT = 27015
args = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ip', default=HOST, type=str)
    parser.add_argument('-p', '--port', default=PORT, type=int)

    return parser.parse_args()


def main():
    global args
    args = get_args()

    print("DBG:Play: starting server")
    print(tools.str_conf())

    server = Server(args.ip, args.port)
    # loading config file
    config = tools.get_param_conf()

    param_file = tools.get_params()
    print("Running with parameters from: {}".format(param_file))
    jit_model_name = ""
    if os.path.isfile(param_file) and config.USE_NN:
        #loading parameters
        print("Loading model parameters from {}".format(param_file))
        params = torch.load(param_file, map_location='cpu')
        #creating instance of model
        model_module = import_module(config.MODEL_MODULE)
        model_class = getattr(model_module, config.MODEL_CLASS)
        model = model_class(config)
        model.load_state_dict(params['state_dict'])
        print("Parameters from: {}".format(param_file))
        jit_model_name = "{}/tmp_{}.pt".format(
                os.path.dirname(os.path.realpath(__file__)),
                os.path.basename(param_file).replace(".pyt",''))
        example = torch.rand(1,2,SHAPE,SHAPE)
        model.eval()

        # compiling network
        with torch.no_grad():
            traced_script_module = torch.jit.trace(model, example)

        traced_script_module.save(jit_model_name)
    elif config.USE_NN:
        print("No parameters found{}".format(param_file))
        return False

    # creating and configureing MCTS
    mcts = cmcts.mcts()
    mcts.set_alpha_default()
    mcts.set_threads(THREADS)
    if config.USE_NN:
        mcts.set_params(jit_model_name)
    print(mcts)

    cnt = 0
    while True:
        print("DBG:Play: waiting for connection")
        server.connect()

        while True:
            cmd, move = server.get_cmd()

            if cmd == cmds.OP_MOVE:
                print("Opponents move")
                sys.stdout.flush()
                mcts.make_movexy(x = move[0], y = move[1])
            elif cmd == cmds.MAKE_MOVE:
                sys.stdout.flush()
                mcts.simulate(sims=config.SIMS, timeout=config.TIMEOUT)
                pi = mcts.get_prob()
                #pi = mcts.heur().reshape(2,-1)
                move = np.argmax(pi)
                mcts.make_move(move)
                server.make_move((move%SHAPE,move//SHAPE))
            elif cmd == cmds.LOAD_MOVE:
                print("Loading move")
                sys.stdout.flush()
                raise NotImplementedError
                #game.set_player()
                #game.make_move(move)
                #game.set_player(1)
            elif cmd == cmds.INITIALIZE:
                print("Initializing")
                sys.stdout.flush()
                print(move)
                #assert move == SHAPE
                np.save("game{}".format(cnt), mcts.get_board())
                cnt += 1
                mcts.clear()
            elif cmd == cmds.END:
                print("End")
                sys.stdout.flush()
                break
            else:
                print('Error: unknow command: '+cmd)
                sys.stdout.flush()
                raise NotImplementedError


    return

if __name__ == "__main__":
    main()
