#ifndef __CMCTS_H__
#define __CMCTS_H__

#include "node.h"
#include <pybind11/numpy.h>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

struct Cmcts{
public:
	void make_move(int action);      // change root_node by taking action
	void print_node(std::vector<int> &v);
	void clear(void);                // restore to initial node
	void simulate(int n);            // run n searches
	void add_predictor(std::function<py::tuple(py::array_t<float>, py::object)> &p, py::dict &d);
	float is_end();               // check if game is over
	std::string repr();
	py::array_t<float> get_prob();
	py::array_t<float> nn_input();
	/* initialization function */
	Cmcts(int seed);
	~Cmcts(void);

#ifdef HEUR
	void print_heur();
#endif
private:
	void search(void);               // run one search starting from initial node

	Node* root_node = nullptr; // game node
	int root_player;
	int root_move_cnt;
	Board root_board;
	double *alpha;
	gsl_rng *r;

	std::function<py::tuple(py::array_t<float>, py::dict)> predict;     // predict function
	py::dict data;       // data passed to predict function
	int player;
	int move_cnt;
	Board board;

#ifdef HEUR
	float rollout();
	void  update(int action);
	std::array<double,2*SIZE> hboard;
	std::array<double,2*SIZE> root_hboard;
#endif
};

#endif
