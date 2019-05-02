#ifndef __STATE_H__
#define __STATE_H__

#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <math.h>
#include <string>
#include <memory>
//#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <thread>

namespace py = pybind11;

using Board = std::array<char, SIZE*2>;

struct State{
	State();
	State(const State *obj);
	~State();

	float is_end();
	bool  is_valid(int action);
	void  clear();
	void  clear(const State *obj);
	void  make_move(int action);
	py::array_t<float> get_board();
	std::string repr();
#ifdef HEUR
	void print_heur();
	void rollout_move();
	void update(int action);
	std::array<double,2*SIZE> hboard;
#endif

	Board board;
	int player;
	int move_cnt;
	float winner;
	std::vector<int> moves;
};

#endif
