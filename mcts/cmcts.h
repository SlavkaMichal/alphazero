#ifndef __CMCTS_H__
#define __CMCTS_H__

#include "node.h"
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

struct Cmcts{
public:
	void make_move(int action);      // change root_node by taking action
	void make_move(int y, int x);
	void print_node(std::vector<int> &v);
	void print_u(std::vector<int> &v);
	void clear(void);                // restore to initial node
	void clear_predictor();
	void simulate(int n);            // run n searches
	void set_predictor(std::function<py::tuple(py::array_t<float>, py::object)> &p, py::object &d);
	void set_seed(unsigned long int seed);
	void set_alpha(double alpha);
	void set_cpuct(float cpuct);
	/* use of this function is depriciated use get_winner instead */
	float is_end();               // check if game is over
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
#endif
private:
	void search(void);               // run one search starting from initial node
	void board_move(int action);

	Node* root_node = nullptr; // game node
	double *alpha;
	double *dir_noise;
	double cpuct;
	gsl_rng *r;
	std::function<py::tuple(py::array_t<float>, py::object)> predict;     // predict function
	py::object data;       // data passed to predict function

	// TODO zbavit sa tychto veci, nemali by byt sucastou objektu
	Board board;
	int player;
	int move_cnt;
	float winner;

#ifdef HEUR
	float rollout();
	void  update(int action);
	std::array<double,2*SIZE> hboard;
#endif
};

#endif
