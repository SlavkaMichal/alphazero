#!/usr/bin/python
import sys
sys.path.append('..')
import site
from config import *
import numpy as np
import cmcts
import argparse
import server as cmds
import tools
import torch
from tools import rand_uint32
from server import Server
from importlib import import_module
model_module = import_module(MODEL_MODULE)

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
    server = Server(args.ip, args.port)
    mcts = cmcts.mcts()
    param_file = tools.get_params()
    params = torch.load(param_file)
    model_class = getattr(model_module, MODEL_CLASS)
    model = model_class()
    model.load_state_dict(params['state_dict'])
    print(param_file)
    jit_model_name = "{}/tmp_{}.pt".format(
            os.path.dirname(os.path.realpath(__file__)),
            os.path.basename(param_file).replace(".pyt",''))
    example = torch.rand(1,2,SHAPE,SHAPE)
    model.eval()
    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, example)

    traced_script_module.save(jit_model_name)
    mcts = cmcts.mcts(seed=rand_uint32(), cpuct=CPUCT)
    mcts.set_alpha_default()
    mcts.set_threads(THREADS)
    mcts.set_params(jit_model_name)

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
                print("My move sims {}".format(SIMS))
                sys.stdout.flush()
                mcts.simulate(SIMS)
                print("get")
                pi = mcts.get_prob()
                mcts.print_node([])
                move = np.argmax(pi)
                mcts.make_move(move)
                print("made move")
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
