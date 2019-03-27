#!/usr/bin/python
# server

import socket
import sys

OP_MOVE   = 1
MAKE_MOVE = 2
LOAD_MOVE = 3
INITIALIZE= 4
END       = 5

class Server:
    def __init__(self, ip, port):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
            self.sock.bind((ip,port))
            self.sock.listen()
        except socket.error as err:
            print("DBG:Server: Error while creating socket: {0}".format(err))
            self.sock.close()

    def __del__(self):
        print("DBG:Server: socked closed")
        self.sock.close()

    def connect(self):
        self.connection, caddr = self.sock.accept()
        print("DBG:Server: Connected to "+caddr[0]+":"+str(caddr[1]))

    def make_move(self, move):
        self.connection.sendall(bytearray(move))
        print("DBG:Server: Making move x:"+str(move[0])+" y:"+str(move[1]))

    def get_cmd(self):
        print("DBG:Server: get_cmd()")
        data = self.connection.recv(32)
        if len(data) == 0:
            print("DBG:Server: Game is over ")
            return END, None

        cmd = data[0]
        print("DBG:Server: '"+str(data)+"'")
        sys.stdout.flush()

        if cmd == LOAD_MOVE:
            self.set_player = True
            return cmd, (data[1], data[2], data[3])
        if cmd == OP_MOVE:
            print("DBG:Server: Oponents move x:"+str(data[1])+" y:"+str(data[2]))
            return cmd, (data[1], data[2])
        if cmd == INITIALIZE:
            return cmd, data[1]

        return cmd, None

