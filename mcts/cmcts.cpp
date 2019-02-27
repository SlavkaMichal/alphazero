#include <algorithm>
#include <stack>
#include <array>
#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include <math.h>
#include "cmcts.h"

namespace py = pybind11;

Cmcts::Cmcts(int seed) :
	root_player(0),
	root_move_cnt(0),
	player(0),
	move_cnt(0)
{
	root_node = new Node();
	const gsl_rng_type *T;
	gsl_rng_env_setup();
	std::fill(root_board.data(), root_board.data()+2*SIZE, 0);
	board = root_board;
	player =  root_player;

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
#ifdef RANDOM
	gsl_rng_set(r, time(NULL)+seed);
#endif
#ifdef HEUR
	std::fill(root_hboard.data(), root_hboard.data()+2*SIZE, 1);
	root_hboard[SIZE/2] = 2;
#endif
	alpha = new double[SIZE];
	// x = avg_game_length = SHAPE*2
	// 10/((SIZE*x-(x**2+x)*0.5)/x)
	double len = SHAPE *2; // average game lenght estimate
	double num = (SIZE*len - (len*len+len)*0.5)/len;
	std::fill(alpha, alpha+SIZE, 10/num);
}

Cmcts::~Cmcts(void)
{
	std::cout << "destructing root_node" << std::endl;
	delete root_node;
	std::cout << "finnished root_node" << std::endl;
	gsl_rng_free(r);
	delete[] alpha;
}

void
Cmcts::clear(void)
{
	delete root_node;
	root_node = new Node();
	std::fill(root_board.data(), root_board.data()+2*SIZE, 0);
	root_player = 0;
	root_move_cnt = 0;
	std::fill(board.data(), board.data()+2*SIZE, 0);
	player = 0;
	move_cnt = 0;

#ifdef HEUR
	std::fill(root_hboard.data(), root_hboard.data()+2*SIZE, 1);
#endif
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
		move_cnt = root_move_cnt;
#ifdef HEUR
		hboard = root_hboard;
#endif
		search();
	}

	board  = root_board;
	player = root_player;
	move_cnt = root_move_cnt;
#ifdef HEUR
	hboard = root_hboard;
#endif

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

	/* TODO
	   is_end vrati -1 ak player prehral 1 ak player vyhral
	   0 ak hra pokracuje
	   */
	while ((value = is_end()) == 0){
		/* new expanded node */
		if (current->nodeN == -1){

			/* simulation */
			if (predict != nullptr){
				gsl_ran_dirichlet(r, SIZE, alpha, dir);
				prediction_t = predict(nn_input(), data);
				value = -prediction_t[0].cast<float>();
				current->set_prior(prediction_t[1].cast<py::array_t<float>>(), dir);
			}else{
#ifdef HEUR
				value = rollout();
				current->set_prior(hboard);
#else
				throw std::runtime_error("Predictor missing!");
#endif
			}
			break;
		}

		nodes.push(current);
		/* select new node, if it's leaf then expand */
		action = current->select(board);
		current = current->next_node(action);
		board[player*SIZE + action] = 1;
#ifdef HEUR
		update(action);
#endif
		player = player ? 0 : 1;
		move_cnt += 1;
	}

	/* backpropagate value */
	while (!nodes.empty()){
		current = nodes.top();
		nodes.pop();

		current->backpropagate(value);
		value = -value;
	}

	delete[] dir;
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
	if (root_board[action] != 0 or root_board[SIZE+action] != 0){
		throw std::runtime_error("Invalid move!");
	}

	Node *old_node = root_node;

	root_node = root_node->make_move(action);
	root_board[root_player*SIZE + action] = 1;
	root_player = root_player ? 0 : 1;
	root_move_cnt += 1;

	board = root_board;
	move_cnt = root_move_cnt;
#ifdef HEUR
	// update root_hboard
	hboard = root_hboard;
	update(action);
	root_hboard = hboard;
#endif
	player = root_player;
	board  = root_board;

	delete old_node;
	return;
}

py::array_t<float>
Cmcts::nn_input()
{
	auto b = new std::vector<float>(board.begin(),board.end());
	for (size_t i = 0; i < b->size(); i++){
		std::cout << b->at(i) << ":" <<board[i] << " ";
	}
	if (player == 1)
		std::swap_ranges(b->begin(), b->begin()+SIZE, b->begin() + SIZE);

	auto capsule = py::capsule(b, [](void *b) { delete reinterpret_cast<std::vector<float>*>(b);});

	return py::array_t<float>(std::vector<ptrdiff_t>{1,2,SHAPE,SHAPE}, b->data(), capsule);
}

float
Cmcts::is_end()
{
	int v0 = 0, h0 = 0, lr0 = 0, rl0 = 0,
	    v1 = 0, h1 = 0, lr1 = 0, rl1 = 0;

	if (move_cnt < 9)
		return 0;

	/* checking vertical and horizontal row of 5 */
	for (int j = 0; j < SHAPE; j++){
		// sum of first five horizontal elements for paler 1
		// I hope that compiler will rollout this loop
		h0 = h1 = v0 = v1 = 0;
		for (int i = 0; i < 5; i++){
			h0 += board[j*SHAPE+i];
			h1 += board[j*SHAPE+i+SIZE];
			v0 += board[i*SHAPE+j];
			v1 += board[i*SHAPE+j+SIZE];
		}
		if (h0 == 5 || v0 == 5){
			return player == 0 ? 1. : -1.;
		}
		if (h1 == 5 || v1 == 5){
			return player == 1 ? 1. : -1.;
		}

		/* subtracting first element and addig one after last so the sum won't be recomputed */
		for (int i = 0; i < SHAPE - 5; i++){
			h0 = h0 - board[j*SHAPE+i]      + board[j*SHAPE+i+5];
			h1 = h1 - board[j*SHAPE+i+SIZE] + board[j*SHAPE+i+5+SIZE];
			v0 = v0 - board[i*SHAPE+j]      + board[(i+5)*SHAPE+j];
			v1 = v1 - board[i*SHAPE+j+SIZE] + board[(i+5)*SHAPE+j+SIZE];

			if (h0 == 5 || v0 == 5){
				return player == 0 ? 1. : -1.;
			}
			if (h1 == 5 || v1 == 5){
				return player == 1 ? 1. : -1.;
			}
		}
	}

	/* checking diagonals */
	/* winning length can fit into SHAPE - (WIN_LEN-1) diagonals */
	for (int j = 0; j < SHAPE - 4; j++){
		/*
		   starting from main diagonal moving in x direction
		   lr - left to right diagonal
		   rl - right to left diagonal
		   sum of first elements in diagonal taken from j-th index
		   the same is done for flipped board
		*/
		lr0 = lr1 = rl0 = rl1 = 0;
		for (int i = 0; i < 5; i++){
			lr0 += board[i*SHAPE+i+j];
			lr1 += board[i*SHAPE+i+j+SIZE];
			rl0 += board[i*SHAPE+SHAPE-1-i-j];
			rl1 += board[i*SHAPE+SHAPE-1-i-j+SIZE];
		}
		if (lr0 == 5 || rl0 == 5){
			return player == 0 ? 1. : -1.;
		}
		if (lr1 == 5 || rl1 == 5){
			return player == 1 ? 1. : -1.;
		}

		for (int i = 0; i < SHAPE - 5 - j; i++){
			lr0 = lr0 - board[i*SHAPE+i+j]      + board[(i+5)*SHAPE+i+j+5];
			lr1 = lr1 - board[i*SHAPE+i+j+SIZE] + board[(i+5)*SHAPE+i+j+5+SIZE];
			rl0 = rl0 - board[i*SHAPE+SHAPE-1-i-j]      + board[(i+5)*SHAPE+SHAPE-1-i-j-5];
			rl1 = rl1 - board[i*SHAPE+SHAPE-1-i-j+SIZE] + board[(i+5)*SHAPE+SHAPE-1-i-j-5+SIZE];

			if (lr0 == 5 || rl0 == 5){
				return player == 0 ? 1. : -1.;
			}
			if (lr1 == 5 || rl1 == 5){
				return player == 1 ? 1. : -1.;
			}
		}
		if (j == 0)
			continue;

		/*
		   the same as above but moving in y direction
		   starts from one to exclude main diagonal which was computed above
		*/
		lr0 = lr1 = rl0 = rl1 = 0;
		for (int i = 0; i < 5; i++){
			lr0 += board[(i+j)*SHAPE+i];
			lr1 += board[(i+j)*SHAPE+i+SIZE];
			rl0 += board[(SHAPE-1-i-j)*SHAPE+i];
			rl1 += board[(SHAPE-1-i-j)*SHAPE+i+SIZE];
		}

		if (lr0 == 5 || rl0 == 5){
			return player == 0 ? 1. : -1.;
		}
		if (lr1 == 5 || rl1 == 5){
			return player == 1 ? 1. : -1.;
		}

		for (int i = 0; i < SHAPE - 5 - j; i++){
			lr0 = lr0 - board[(i+j)*SHAPE+i]      + board[(i+j+5)*SHAPE+i+5];
			lr1 = lr1 - board[(i+j)*SHAPE+i+SIZE] + board[(i+j+5)*SHAPE+i+5+SIZE];
			rl0 = rl0 - board[(SHAPE-1-i-j)*SHAPE+i]      + board[(SHAPE-1-i-j-5)*SHAPE+i+5];
			rl1 = rl1 - board[(SHAPE-1-i-j)*SHAPE+i+SIZE] + board[(SHAPE-1-i-j-5)*SHAPE+i+5+SIZE];

			if (lr0 == 5 || rl0 == 5){
				return player == 0 ? 1. : -1.;
			}
			if (lr1 == 5 || rl1 == 5){
				return player == 1 ? 1. : -1.;
			}
		}
	}

	if (move_cnt == SIZE)
		return 1e-8;
	return 0.;
}

#ifdef HEUR
float
Cmcts::rollout()
{
	int action = 0;
	float v = 0;
	double max = 0;
	double h = 0;
	/* save original state */
	std::array<double,2*SIZE> start_hboard = hboard;
	Board start_board = board;
	int start_player = player;
	int start_cnt = move_cnt;

	while ((v = is_end()) == 0){
		/* choose action */
		for (int a = 0; a < SIZE; a++){
			if (board[a] == 0 and board[SIZE+a] == 0){
				h = hboard[a]+hboard[SIZE+a];
				if (max < h){
					max = h;
					action = a;
				}
			}
		}
		if (max == -1){
			break;
		}
		max = -1;

		board[player*SIZE + action] = 1;
		update(action);
		player = player ? 0 : 1;
		move_cnt++;
		if (move_cnt >= SIZE)
			break;
	}

	v = start_player == player ? v : -v;
	/* restore original state */
	hboard = start_hboard;
	move_cnt = start_cnt;
	board = start_board;
	player = start_player;
	return v;
}

void
Cmcts::update(int action)
{
	int l, r;
	int lbound, rbound;
	int x, y, tmp;
	int op = player == 1 ? 0 : 1;
	int reward;

	y = action/SHAPE;
	x = action%SHAPE;
	hboard[SIZE+action] = 0;
	hboard[action] = 0;
	// horizontal -
	l = action-1;
	r = action+1;
	lbound = l;
	rbound = r;

	tmp = (y+1)*SHAPE;
	while (tmp > rbound && !board[op*SIZE+rbound]) ++rbound;
	tmp = y*SHAPE-1;
	while (tmp < lbound && !board[op*SIZE+lbound]) --lbound;

	if (rbound-lbound > 5){
		while (lbound < l && board[player*SIZE+l]) --l;
		while (rbound > r && board[player*SIZE+r]) ++r;
		reward = r-l-1;
		if (lbound != l && (hboard[l]||hboard[SIZE+l])){
			hboard[player*SIZE+l] += std::pow(4, reward);
//			if (action-l-1 != 0)
//				hboard[player*SIZE+l] -= std::pow(4, action-l-1);
		}
		if (rbound != r && (hboard[r]||hboard[SIZE+r])){
			hboard[player*SIZE+r] += std::pow(4, reward);
//			if (r-action-1 != 0)
//				hboard[player*SIZE+r] -= std::pow(4, r-action-1);
		}

	}
	else // you can't make 5 in row here
		while (++lbound < rbound) hboard[player*SIZE+lbound] = 0;

	// vertical |
	l = action-SHAPE;
	r = action+SHAPE;
	lbound = l;
	rbound = r;

	tmp = x-SHAPE;
	while (tmp < lbound && !board[op*SIZE+lbound]) lbound -= SHAPE;
	tmp = x+SIZE;
	while (tmp > rbound && !board[op*SIZE+rbound]) rbound += SHAPE;

	if (rbound-lbound > 5*SHAPE){
		while (lbound < l && board[player*SIZE+l]) l -= SHAPE;
		while (rbound > r && board[player*SIZE+r]) r += SHAPE;
		// y coordinate of bottom(r) minus y coordinate of top(l)
		// vertical distance between r and l
		reward = r/SHAPE-l/SHAPE-1;
		if (lbound != l && (hboard[l]||hboard[SIZE+l])){
			hboard[player*SIZE+l] += std::pow(4, reward);
//			if (action-l-SHAPE != 0)
//				hboard[player*SIZE+l] += std::pow(4, action/SHAPE-l/SHAPE-1);
		}
		if (rbound != r && (hboard[r]||hboard[SIZE+r])){
			hboard[player*SIZE+r] += std::pow(4, reward);
//			if (r-action-SHAPE != 0)
//				hboard[player*SIZE+r] += std::pow(4, r/SHAPE-action/SHAPE-1);
		}

	}
	else{ // you can't make 5 in row here
		lbound += SHAPE;
		while (lbound < rbound){
		       hboard[player*SIZE+lbound] = 0;
		       lbound += SHAPE;
		}
	}

	/* lr diagonal \ */
	l = action-SHAPE-1;
	r = action+SHAPE+1;
	lbound = l;
	rbound = r;

	tmp = x > y ? action-(SHAPE+1)*(y+1) : action-(SHAPE+1)*(x+1);
	while (tmp < lbound && !board[op*SIZE+lbound]) lbound -= (SHAPE+1);
	tmp = x > y ? action+(SHAPE-x)*(SHAPE+1) : action+(SHAPE-y)*(SHAPE+1);
	while (tmp > rbound && !board[op*SIZE+rbound]) rbound += (SHAPE+1);

	if (rbound/SHAPE-lbound/SHAPE > 5){
		while (lbound < l && board[player*SIZE+l]) l -= (SHAPE+1);
		while (rbound > r && board[player*SIZE+r]) r += (SHAPE+1);
		// y coordinate of bottom(r) minus y coordinate of top(l)
		// vertical distance between r and l
		reward = r/SHAPE-l/SHAPE-1;
		if (lbound != l && (hboard[l]||hboard[SIZE+l])){
			hboard[player*SIZE+l] += std::pow(4, reward);
//			if (action/SHAPE-l/SHAPE-1 != 0)
//				hboard[player*SIZE+l] += std::pow(4, action/SHAPE-l/SHAPE-1);
		}
		if (rbound != r && (hboard[r]||hboard[SIZE+r])){
			hboard[player*SIZE+r] += std::pow(4, reward);
//			if (r/SHAPE-action/SHAPE-1 != 0)
//				hboard[player*SIZE+r] += std::pow(4, r/SHAPE-action/SHAPE-1);
		}

	}
	else{ // you can't make 5 in row here
		lbound += SHAPE + 1;
		while (lbound < rbound){
		       hboard[player*SIZE+lbound] = 0;
		       lbound += SHAPE + 1;
		}
	}

	/* lr diagonal / */
	l = action-SHAPE+1;
	r = action+SHAPE-1;
	lbound = l;
	rbound = r;

	tmp = x > y ? action-(SHAPE-1)*(y+1) : action-(SHAPE-1)*(x+1);
	while (tmp < lbound && !board[op*SIZE+lbound]) lbound -= (SHAPE-1);
	tmp = x > y ? action+(SHAPE-x+1)*(SHAPE-1) : action+(SHAPE-y+1)*(SHAPE-1);
	while (tmp > rbound && !board[op*SIZE+rbound]) rbound += (SHAPE-1);


	if (rbound/SHAPE-lbound/SHAPE > 5){
		while (lbound < l && board[player*SIZE+l]) l -= (SHAPE-1);
		while (rbound > r && board[player*SIZE+r]) r += (SHAPE-1);
		// y coordinate of bottom(r) minus y coordinate of top(l)
		// vertical distance between r and l
		reward = r/SHAPE-l/SHAPE-1;
		if (lbound != l && (hboard[l]||hboard[SIZE+l])){
			hboard[player*SIZE+l] += std::pow(4, reward);
//			if (action/SHAPE-l/SHAPE-1 != 0)
//				hboard[player*SIZE+l] += std::pow(4, action/SHAPE-l/SHAPE-1);
		}
		if (rbound != r && (hboard[r]||hboard[SIZE+r])){
			hboard[player*SIZE+r] += std::pow(4, reward);
//			if (r/SHAPE-action/SHAPE-1 != 0)
//				hboard[player*SIZE+r] += std::pow(4, r/SHAPE-action/SHAPE-1);
		}

	}
	else{ // you can't make 5 in row here
		lbound += SHAPE - 1;
		while (lbound < rbound){
		       hboard[player*SIZE+lbound] = 0;
		       lbound += SHAPE - 1;
		}
	}
	return;
}

void
Cmcts::print_heur()
{
	for (int i = 0; i < SHAPE; i++){
		for (int j = 0; j < SHAPE; j++){
			std::cout << root_hboard[SIZE+i*SHAPE+j]+root_hboard[i*SHAPE+j] << " ";
		}
		std::cout << std::endl;
	}
	return;
}

#endif

std::string
Cmcts::repr()
{
	std::string s;
	float winner;
	s.append("First player: x\n");
	if (root_player == 0)
		s.append("Player: x\n");
	else
		s.append("Player: o\n");
	winner = is_end();
	if (winner == 0)
		s.append("Winner: _\n");
	else if ((winner == 1 && root_player == 0) || (winner == -1 && root_player == 1))
		s.append("Winner: x\n");
	else if ((winner == 1 && root_player == 1) || (winner == -1 && root_player == 0))
		s.append("Winner: o\n");
	else
		s.append("Winner: draw\n");

	s.append("Alpha: "+std::to_string(alpha[0])+"\n");
	if (predict == nullptr)
		s.append("Heuristic: yes\n");
	else
		s.append("Heuristic: no\n");

	s.append("Move count: "+std::to_string(root_move_cnt)+"\n");
	s.append("Board size: "+std::to_string(SIZE)+"\n");
	s.append("Board:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++)
			if (root_board[i*SHAPE+j] == 1)
				s.append("x ");
			else if (root_board[i*SHAPE+j+SIZE] == 1)
				s.append("o ");
			else
				s.append("_ ");
		s.append("\n");
	}

	return s;
}

void
Cmcts::print_node(std::vector<int> &v)
{
	Node *n = root_node;
	for (int i : v){
		if (n->children[i] == nullptr){
			py::print("child is nullptr");
			return;
		}
		n = n->next_node(i);
		std::cout << i << std::endl;
	}
	py::print(n->repr());
	return;
}
