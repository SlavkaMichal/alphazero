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
#include <memory>

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
	bool is_null(int a);
	int nodeN;

	Node(std::string &name, int a);
	~Node();
private:
	std::string name;
	struct Node* children[SIZE];
	std::array<std::unique_ptr<Node>, SIZE> child;
	int last_action;
	int child_cnt;
	std::array<int, SIZE> edgeN;
	float edgeP[SIZE];
	float edgeW[SIZE];
};

#endif
