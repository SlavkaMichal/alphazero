#ifndef __CMCTS_H__
#define __CMCTS_H__

#include "node.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

struct Cmcts{
public:
	void make_move(int action);      // change root_node by taking action
	int  get_result();               // check if game is over
	//std::string repr();
	void clear(void);                // restore to initial node
	void simulate(int n);            // run n searches
	py::array_t<float> get_prob();
	py::array_t<float> nn_input();
	void add_predictor(std::function<py::tuple(py::array_t<float>, py::object)> &p, py::dict &d);
	/* initialization function */
	Cmcts(void);
	~Cmcts(void);
private:
	void search(void);               // run one search starting from initial node

	Node* root_node = nullptr; // game node
	int root_player;
	Board root_board;
	gsl_rng *r;

	std::function<py::tuple(py::array_t<float>, py::dict)> predict;     // predict function
	py::object data;       // data passed to predict function
	int player;
	Board board;
};

#endif
