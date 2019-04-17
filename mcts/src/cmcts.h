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
#include <torch/csrc/api/include/torch/utils.h>
#include <thread>

struct Cmcts{
public:
	Cmcts(uint64_t seed, double alpha, double cpuct);
	~Cmcts(void);

	void make_move(int action);      // change root_node by taking action
	void make_movexy(int x, int y);

	void clear(void);                // restore to initial node
	void clear_params();
	void simulate(int n);            // run n searches

	/* getters and setters */
	void              set_params(std::string &file_name);
	const std::string get_params() const;
	void              set_threads(int threads);
	const int         get_threads() const;
	void              set_cuda(int cuda);
	const int         get_cuda() const;
	void              set_alpha(double alpha);
	const double      get_alpha() const;
	void              set_player(int player);
	const int         get_player() const;
	void              set_cpuct(float cpuct);
	const float       get_cpuct() const;
	void              set_seed(unsigned long int seed);
	void              set_alpha_default();
	const float       get_winner() const;
	const int         get_move_cnt() const;

	py::array_t<float> get_prob();
	py::array_t<float> get_board();
	std::string repr();

	/* for debugging purposes */
	void print_node(std::vector<int> &v);
	void print_u(std::vector<int> &v);

#ifdef HEUR
	py::array_t<float> get_heur();
	void print_heur();
	float rollout();
#endif
private:
	void worker(int n);               // run one search starting from initial node
	void search(State *state, std::shared_ptr<torch::jit::script::Module> module);               // run one search starting from initial node

	Node   *root_node = nullptr; // game node
	State  *state;
	double *alpha;     // constant for generating dirichlet noise
	double *dir_noise; // array alocated for dirichlet noise
	double cpuct;      // constant affecting how much MCTS relies on neural network
	gsl_rng *r;
	std::string param_name; // file from which parameters will be loaded
	int threads;            // number of threads
	int cuda;            // number of threads
};

#endif
