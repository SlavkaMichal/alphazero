#ifndef __NODE_H__
#define __NODE_H__

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cmath>
#include <math.h>

namespace py = pybind11;
using Board = std::array<int, SIZE*2>;

struct Node{
public:
	int select(Board &board);
	struct Node* next_node(int action);
	void backpropagate(float value);
	void set_prior(py::array_t<float> p, double* dir);
	std::array<int, SIZE>* counts();
	int nodeN;

	Node();
	~Node();
private:
	std::array<int, SIZE> edgeN;
	float edgeW[SIZE];
	float edgeP[SIZE];
	struct Node* children[SIZE];
	int   last_action;
};

#endif
