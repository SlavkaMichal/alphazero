#ifndef __NODE_H__
#define __NODE_H__

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cmath>
#include <math.h>
#include <string>

namespace py = pybind11;
using Board = std::array<int, SIZE*2>;

struct Node{
public:
	int select(Board &board);
	struct Node* next_node(int action);
	struct Node* make_move(int action);
	void backpropagate(float value);
	void set_prior(py::array_t<float> p, double* dir);
	void set_prior(std::array<double, 2*SIZE> &hboard);
	std::array<int, SIZE>* counts();
	std::string repr();
	int nodeN;
	int child_cnt;
	int last_action;
	float edgeP[SIZE];
	struct Node* children[SIZE];

	Node();
	~Node();
private:
	std::array<int, SIZE> edgeN;
	float edgeW[SIZE];
};

#endif
