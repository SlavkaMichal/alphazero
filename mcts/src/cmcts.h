#ifndef __CMCTS_H__
#define __CMCTS_H__

#include "node.h"
#include "state.h"
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <stack>
#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include <math.h>
#include <iostream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <torch/script.h>
#include <thread>

struct Cmcts{
public:
	void make_move(int action);      // change root_node by taking action
	void make_movexy(int x, int y);
	void print_node(std::vector<int> &v);
	void print_u(std::vector<int> &v);
	void clear(void);                // restore to initial node
	void clear_params();
	void simulate(int n);            // run n searches
	void set_params(std::string &file_name);
	void set_seed(unsigned long int seed);
	void set_alpha(double alpha);
	void set_alpha_default();
	void set_cpuct(float cpuct);
	/* use of this function is depriciated use get_winner instead */
	float get_winner();
	int   get_player();
	int   get_move_cnt();
	py::array_t<float> get_prob();
	py::array_t<float> get_board();
	/* initialization function */
	Cmcts(uint64_t seed, double alpha, double cpuct);
	~Cmcts(void);
	std::string repr();

#ifdef HEUR
	py::array_t<float> get_heur();
	void print_heur();
	float rollout();
#endif
private:
	void worker(int n);               // run one search starting from initial node
	void search(State *state, std::shared_ptr<torch::jit::script::Module> module);               // run one search starting from initial node
	//void board_move(int action);

	Node   *root_node = nullptr; // game node
	State  *state;
	double *alpha;
	double *dir_noise;
	double cpuct;
	gsl_rng *r;
	std::string param_name;
	//std::function<py::tuple(py::array_t<float>, py::object)> predict;     // predict function
	//py::object data;       // data passed to predict function
};

#endif
