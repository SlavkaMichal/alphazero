#include <algorithm>
#include <stack>
#include <array>
#include <vector>
#include <cmath>
#include <math.h>
#include "cmcts.h"

namespace py = pybind11;

Cmcts::Cmcts() :
	root_player(0),
	root_board{}
{
	root_node = new Node();
	const gsl_rng_type *T;
	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);


}

Cmcts::~Cmcts(void)
{
	delete root_node;
	gsl_rng_free(r);
	//delete[] root_board;
}

//std::string
//Cmcts::repr()
//{
//
//}

void
Cmcts::clear(void)
{
	delete root_node;
	root_node = new Node();
	std::fill(root_board.data(), root_board.data()+2*SIZE, 0);
	//root_board.fill(0);
	root_player = 1;
	return;
}

void
Cmcts::add_predictor(std::function<py::tuple(py::array_t<float>, py::object)> &p, py::dict &d)
{
	predict = p;
	data    = d;
	return;
}


void
Cmcts::simulate(int n)
{
	// TODO skontrolovat ci toto kopiruje
	for (int i = 0; i < n; ++i){
		board  = root_board;
		player = root_player;
		search();
	}

	board  = root_board;
	player = root_player;

	return;
}

void
Cmcts::search(void)
{
	Node* current = root_node;
	std::stack<Node*> nodes;
	float value = 0;
	int action;

	py::tuple prediction_t;
	auto dir = new double[SIZE];
	auto alpha = new double[SIZE];
	std::fill(alpha, alpha+SIZE, 0.03);

	while ((value == get_result()) != 0){
		/* new expanded node */
		if (current->nodeN == -1){
			/* simulation */
			if (predict){
				prediction_t = predict(nn_input(), data);
				value = -prediction_t[0].cast<float>();
				gsl_ran_dirichlet(r, SIZE, alpha, dir);
				current->set_prior(prediction_t[1].cast<py::array_t<float>>(), dir);
			}
			else{
				value = 1;
			}
			break;
		}

		nodes.push(current);
		/* select new node, if it's leaf then expand */
		action = current->select(board);
		current = current->next_node(action);
		board[player*SIZE + action] = 1;
		if (player)
			player = 0;
		else
			player = 1;
	}

	/* backpropagate value */
	while (!nodes.empty()){
		current = nodes.top();
		nodes.pop();

		current->backpropagate(value);
		value = -value;
	}

	delete[] dir;
	delete[] alpha;
	return;
}

py::array_t<float>
Cmcts::get_prob()
{
	/* all nonvalid moves should not be played,
	   they are skiped in select phase so visit count should be zero
	 */
	std::array<int, SIZE>* counts = root_node->counts();
	int sum = root_node->nodeN;
	auto v = new std::vector<float>(counts->begin(), counts->end());
	for (int i = 0; i < SIZE; i++)
		v->at(i) = v->at(i) / sum;
	auto capsule = py::capsule(v, [](void *v) { delete reinterpret_cast<std::vector<float>*>(v);});

	return py::array_t<float>(v->size(), v->data(), capsule);
}

void
Cmcts::make_move(int action)
{
	Node *old_node = root_node;
	root_node = root_node->next_node(action);
	delete old_node;
	return;
}

py::array_t<float>
Cmcts::nn_input()
{
	// TODO treba zmenit poradie hracich ploch pre hracov
	//py::array_t<float> nn_board;
	auto b = new std::vector<float>(board.begin(),board.end());
	if (player == 1)
		std::swap_ranges(b->begin(), b->begin()+SIZE, b->begin() + SIZE);

	auto capsule = py::capsule(b, [](void *b) { delete reinterpret_cast<std::vector<float>*>(b);});

	return py::array_t<float>(b->size(), b->data(), capsule);
//	return py::buffer_info(
//			b.data(),
//			sizeof(float),
//			py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
//			3,                                      /* Number of dimensions */
//			std::vector<ssize_t>{ 2, SHAPE, SHAPE },                 /* Buffer dimensions */
//			std::vector<ssize_t>{ sizeof(float) * SIZE,             /* Strides (in bytes) for each index */
//			  sizeof(float) * SHAPE
//			  sizeof(float) }
//			);
}

int
Cmcts::get_result()
{
	int v0 = 0, h0 = 0, lr0 = 0, rl0 = 0,
	    v1 = 0, h1 = 0, lr1 = 0, rl1 = 0;

	/* checking vertical and horizontal row of 5 */
	for (int j = 0; j < SHAPE; j++){
		// sum of first five horizontal elements for paler 1
		// I hope that compiler will rollout this loop
		for (int i = 0; i < 5; i++){
			h0 += board[j*SHAPE+i];
			h1 += board[j*SHAPE+i+SIZE];
			v0 += board[i*SHAPE+j];
			v1 += board[i*SHAPE+j+SIZE];
		}
		/* if winlength was 3 initial summed element would look like this
		   | h and v | h | h |   |   | ...
		   |    v    |   |   |   |   | ...
		   |    v    |   |   |   |   | ...
		   |         |   |   |   |   | ...
		   |         |   |   |   |   | ...
		*/
		// to win every element must be 1, therefore sum is 5
		// checking first five elements
		if (h0 == 5 || v0 == 5)
			return player == 0 ? 1 : -1;
		if (h1 == 5 || v1 == 5)
			return player == 1 ? 1 : -1;

		/* subtracting first element and addig one after last so the sum won't be recomputed */
		for (int i = 0; i < SHAPE - 5; i++){
			h0 = h0 - board[j*SHAPE+i]      + board[j*SHAPE+i+5];
			h1 = h1 - board[j*SHAPE+i+SIZE] + board[j*SHAPE+i+5+SIZE];
			v0 = v0 - board[i*SHAPE+j]      + board[i*SHAPE+j+5+SIZE];
			v1 = v1 - board[i*SHAPE+j+SIZE] + board[i*SHAPE+j+5+SIZE];

			if (h0 == 5 || v0 == 5)
				return player == 0 ? 1 : -1;
			if (h1 == 5 || v1 == 5)
				return player == 1 ? 1 : -1;
		}
	}

	/* checking diagonals */
	for (int j = 0; j < SHAPE - 5; j++){
		/*
		   starting from main diagonal moving in x direction
		   lr - left to right diagonal
		   rl - right to left diagonal
		   sum of first elements in diagonal taken from j-th index
		   the same is done for flipped board
		*/
		for (int i = 0; i < 5; i++){
			lr0 += board[i*SHAPE+i+j];
			lr1 += board[i*SHAPE+i+j+SIZE];
			rl0 += board[i*SHAPE+SHAPE-1-i-j];
			rl1 += board[i*SHAPE+SHAPE-1-i-j+SIZE];
		}

		if (lr0 == 5 || rl0 == 5)
			return player == 0 ? 1 : -1;
		if (lr1 == 5 || rl1 == 5)
			return player == 1 ? 1 : -1;

		for (int i = 0; i < SHAPE - 5 - j; i++){
			lr0 = lr0 - board[i*SHAPE+i+j]      + board[(i+5)*SHAPE+i+j+5];
			lr1 = lr1 - board[i*SHAPE+i+j+SIZE] + board[(i+5)*SHAPE+i+j+5+SIZE];
			rl0 = rl0 - board[i*SHAPE+SHAPE-1-i-j]      + board[(i+5)*SHAPE+SHAPE-1-i-j-5];
			rl1 = rl1 - board[i*SHAPE+SHAPE-1-i-j+SIZE] + board[(i+5)*SHAPE+SHAPE-1-i-j-5+SIZE];

			if (lr0 == 5 || rl0 == 5)
				return player == 0 ? 1 : -1;
			if (lr1 == 5 || rl1 == 5)
				return player == 1 ? 1 : -1;
		}

		/*
		   the same as above but moving in y direction
		   starts from one to exclude main diagonal which was computed above
		*/
		for (int i = 1; i < 5; i++){
			lr0 += board[(i+j)*SHAPE+i];
			lr1 += board[(i+j)*SHAPE+i+SIZE];
			rl0 += board[(SHAPE-1-i-j)*SHAPE+i];
			rl1 += board[(SHAPE-1-i-j)*SHAPE+i+SIZE];
		}

		if (lr0 == 5 || rl0 == 5)
			return player == 0 ? 1 : -1;
		if (lr1 == 5 || rl1 == 5)
			return player == 1 ? 1 : -1;

		for (int i = 0; i < SHAPE - 5 - j; i++){
			lr0 = lr0 - board[(i+j)*SHAPE+i]      + board[(i+j+5)*SHAPE+i+5];
			lr1 = lr1 - board[(i+j)*SHAPE+i+SIZE] + board[(i+j+5)*SHAPE+i+5+SIZE];
			rl0 = rl0 - board[(SHAPE-1-i-j)*SHAPE+i]      + board[(SHAPE-1-i-j-5)*SHAPE+i+5];
			rl1 = rl1 - board[(SHAPE-1-i-j)*SHAPE+i+SIZE] + board[(SHAPE-1-i-j-5)*SHAPE+i+5+SIZE];

			if (lr0 == 5 || rl0 == 5)
				return player == 0 ? 1 : -1;
			if (lr1 == 5 || rl1 == 5)
				return player == 1 ? 1 : -1;
		}
	}

	return 0;
}
