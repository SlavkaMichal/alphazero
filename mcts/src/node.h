#ifndef __NODE_H__
#define __NODE_H__

//#include <pybind11/functional.h>
#include <pybind11/numpy.h>
//#include <pybind11/stl.h>
//#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <pybind11/embed.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <math.h>
#include <string>
#include <memory>
#include <mutex>
#include "state.h"

namespace py = pybind11;

struct Node{
public:
	int select(State *state, double cpuct);
	struct Node* next_node(int action);
	struct Node* make_move(int action);
	void backpropagate(int action, float value);
	void set_prior(torch::Tensor p, double dir_eps);
	void set_prior(State *state, double dir_eps);
	std::array<long int, SIZE>* counts();
	std::string repr();
	std::string print_u(State *state, double cpuct);
	bool is_null(int a);
	long long int nodeN;
	std::mutex node_mutex;
	std::array<double, SIZE> childP;
//	std::string name;

	//Node(std::string &name, int a);
	Node();
	~Node();
private:
	std::array<std::unique_ptr<Node>, SIZE> child;
	int child_cnt;
	std::array<long int, SIZE> childN;
	std::array<double, SIZE> childW;
};

#endif
