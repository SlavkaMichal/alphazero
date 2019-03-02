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
	player(0),
	move_cnt(0),
	winner(-1.)
{
	std::string s = "R";
	root_node = new Node(s,-1);
	const gsl_rng_type *T;
	gsl_rng_env_setup();
	std::fill(board.data(), board.data()+2*SIZE, 0);

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
#ifdef RANDOM
	gsl_rng_set(r, time(NULL)+seed);
#endif
#ifdef HEUR
	std::fill(hboard.data(), hboard.data()+2*SIZE, 0.5);
	hboard[SIZE/2] += 0.5;
	hboard[SIZE+SIZE/2] += 0.5;
#endif
	alpha = new double[SIZE];
	dir_noise = new double[SIZE];
	// x = avg_game_length = SHAPE*2
	// 10/((SIZE*x-(x**2+x)*0.5)/x)
	double len = SHAPE *2; // average game lenght estimate
	double num = (SIZE*len - (len*len+len)*0.5)/len;
	std::fill(alpha, alpha+SIZE, 10/num);
}

Cmcts::~Cmcts(void)
{
	delete root_node;
	gsl_rng_free(r);
	delete[] alpha;
	delete[] dir_noise;
}

void
Cmcts::clear(void)
{
	delete root_node;
	std::string s = "R";
	root_node = new Node(s,-1);
	std::fill(board.data(), board.data()+2*SIZE, 0);
	player = 0;
	move_cnt = 0;
	predict = nullptr;

#ifdef HEUR
	std::fill(hboard.data(), hboard.data()+2*SIZE, 0.5);
	hboard[SIZE/2] += 0.5;
	hboard[SIZE+SIZE/2] += 0.5;
#endif
	return;
}

int
Cmcts::get_player()
{
	return player;
}

int
Cmcts::get_move_cnt()
{
	return move_cnt;
}

float Cmcts::
get_winner()
{
	return winner;
}

void
Cmcts::set_predictor(std::function<py::tuple(py::array_t<float>, py::object)> &p, py::dict &d)
{
	predict = p;
	data    = d;
	return;
}


void
Cmcts::simulate(int n)
{
	// TODO skontrolovat ci toto kopiruje
	if (n < 1)
		throw std::runtime_error("Invalid input "+std::to_string(n)+" must be at leas 1!");
	Board start_board    = board;
	int   start_player   = player;
	int   start_move_cnt = move_cnt;
	float start_winner   = winner;
#ifdef HEUR
	std::array<double,2*SIZE> start_hboard = hboard;
#endif

	for (int i = 0; i < n; ++i){
		search();
		board    = start_board;
		player   = start_player;
		move_cnt = start_move_cnt;
		winner   = start_winner;
#ifdef HEUR
		hboard   = start_hboard;
#endif
	}

#ifdef HEUR
	hboard = start_hboard;
#endif

	return;
}

void
Cmcts::search(void)
{
	Node* current = root_node;
	std::stack<Node*> nodes;
	float value = 0.;
	int action;

	py::tuple prediction_t;

	/* TODO
	   is_end vrati -1 ak player prehral 1 ak player vyhral
	   0 ak hra pokracuje
	   */
	int cnt = 0;
	while (1){
		cnt++;
		if (current->nodeN == -1){
			/* node expansion */
			if (predict != nullptr){
				gsl_ran_dirichlet(r, SIZE, alpha, dir_noise);
				prediction_t = predict(get_board(), data);
				value = -prediction_t[0].cast<float>();
				current->set_prior(prediction_t[1].cast<py::array_t<float>>(), dir_noise);
			}else{
#ifdef HEUR
				value = -rollout();
				current->set_prior(hboard);
#else
				throw std::runtime_error("Predictor missing!");
#endif
			}
			break;
		}

		/* select new node, if it's leaf then expand */
		action = current->select(board);
		board_move(action);
		/* test if it is end state */
		if (get_winner() != -1){
			/* if game is over don't push leaf node */
			/* player on move is the one who lost */
			/* last node on stack is previous and that one also lost */
			if (get_winner() == 0 || get_winner() == 1)
				value = -1.;
			else
				value = get_winner();
			break;
		}

		/* this is not a leaf node nor final node, push it */
		nodes.push(current);
		current = current->next_node(action);
	}
	cnt = 0;

	/* backpropagate value */
	while (!nodes.empty()){
		current = nodes.top();
		cnt++;
		nodes.pop();

		current->backpropagate(value);
		value = -value;
	}
	return;
}


void Cmcts::
board_move(int action)
{
	int end = 0;
	if (action >= SIZE || action < 0)
		throw std::runtime_error("Invalid move "+std::to_string(action)+" out of bound (range is 0-"+std::to_string(SHAPE)+")!");
	if (board[action] != 0 or board[SIZE+action] != 0){
		throw std::runtime_error("Invalid move y: "+std::to_string(action/SHAPE)+", x: "+std::to_string(action%SHAPE)+"!");
	}

	board[player*SIZE + action] = 1;
	player = player ? 0 : 1;
	move_cnt += 1;

	end = is_end();

	if (end != 0.){
		if (end == -1.)
			winner = player == 0 ? 1. : 0.;
		else if (end == 1.)
			winner = player == 1 ? 1. : 0.;
		else
			/* draw */
			winner = 0.5;
	}

#ifdef HEUR
	update(action);
#endif

	return;
}

void
Cmcts::make_move(int action)
{
	Node *old_node = root_node;

	board_move(action);
	root_node = root_node->make_move(action);

	delete old_node;
	return;
}

void
Cmcts::make_move(int y, int x)
{
	if (y >= SHAPE || x >= SHAPE)
		throw std::runtime_error("Invalid index y:"+std::to_string(y)+" or x: "+std::to_string(x)+" (range is 0-"+std::to_string(SIZE)+")!");
	int action = y*SHAPE + x;
	Node *old_node = root_node;

	board_move(action);
	root_node = root_node->make_move(action);

	delete old_node;
	return;
}

py::array_t<float>
Cmcts::get_board()
{
	auto b = new std::vector<float>(board.begin(),board.end());

	if (player == 1)
		std::swap_ranges(b->begin(), b->begin()+SIZE, b->begin() + SIZE);

	auto capsule = py::capsule(b, [](void *b) { delete reinterpret_cast<std::vector<float>*>(b);});

	return py::array_t<float>(std::vector<ptrdiff_t>{2,SHAPE,SHAPE}, b->data(), capsule);
}

py::array_t<float>
Cmcts::get_heur()
{
	auto h = new std::vector<float>(hboard.begin(),hboard.end());

	auto capsule = py::capsule(h, [](void *h) { delete reinterpret_cast<std::vector<float>*>(h);});

	return py::array_t<float>(std::vector<ptrdiff_t>{2,SHAPE,SHAPE}, h->data(), capsule);
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

float
Cmcts::is_end()
{
	int v0 = 0, h0 = 0, lr0 = 0, rl0 = 0,
	    v1 = 0, h1 = 0, lr1 = 0, rl1 = 0;

	if (move_cnt < 9)
		return 0.;

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
	Board start_board  = board;
	int   start_player = player;
	int   start_cnt    = move_cnt;
	float start_winner = winner;

	while (get_winner() == -1){
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

		board_move(action);
	}

	if (get_winner() == 1. || get_winner() == 0.)
		v = start_player == get_winner() ? 1. : -1.;

	/* restore original state */
	hboard   = start_hboard;
	board    = start_board;
	winner   = start_winner;
	move_cnt = start_cnt;
	player   = start_player;
	return v;
}

void
Cmcts::update(int action)
{
	int l, r;
	int lbound, rbound;
	int x, y, tmp;
	int op = this->player == 1 ? SIZE : 0;
	int player = this->player == 1 ? 0 : SIZE;
	int step;
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
	step = 1;

	tmp = y*SHAPE-1;
	while (tmp < lbound && !board[op+lbound]) lbound -= step;
	tmp = (y+1)*SHAPE;
	while (tmp > rbound && !board[op+rbound]) rbound += step;

	while (lbound < l && board[player+l]) l -= step;
	while (rbound > r && board[player+r]) r += step;
	if (rbound-lbound > 5){
		reward = r-l-1;
		if (lbound != l)
			hboard.at(player+l) += std::pow(4, reward);
		if (rbound != r)
			hboard.at(player+r) += std::pow(4, reward);
	}
	if (r-action-step > 0 && rbound != r)
		hboard.at(player+r) -= std::pow(4, r-action-1);
	if (action-l-step > 0 && lbound != l)
		hboard.at(player+l) -= std::pow(4, action-l-1);


	// vertical |
	l = action-SHAPE;
	r = action+SHAPE;
	lbound = l;
	rbound = r;
	step = SHAPE;

	tmp = x-SHAPE;
	while (tmp < lbound && !board[op+lbound]) lbound -= step;
	tmp = x+SIZE;
	while (tmp > rbound && !board[op+rbound]) rbound += step;

	while (lbound < l && board[player+l]) l -= step;
	while (rbound > r && board[player+r]) r += step;
	if (rbound-lbound > 5*SHAPE){
		// y coordinate of bottom(r) minus y coordinate of top(l)
		// vertical distance between r and l
		reward = r/SHAPE-l/SHAPE-1;
		if (lbound != l)
			hboard.at(player+l) += std::pow(4, reward);
		if (rbound != r)
			hboard.at(player+r) += std::pow(4, reward);
	}
	if (r-action-step > 0 && rbound != r)
		hboard.at(player+r) += std::pow(4, r/SHAPE - action/SHAPE-1);
	if (action-l-step > 0 && lbound != l)
		hboard.at(player+l) += std::pow(4, action/SHAPE - l/SHAPE-1);

	/* lr diagonal \ */
	l = action-SHAPE-1;
	r = action+SHAPE+1;
	lbound = l;
	rbound = r;
	step = SHAPE+1;

	tmp = x > y ? action-step*(y+1) : action-step*(x+1);
	while (tmp < lbound && !board[op+lbound]) lbound -= step;
	tmp = x > y ? action+(SHAPE-x)*step : action+(SHAPE-y)*step;
	while (tmp > rbound && !board[op+rbound]) rbound += step;

	while (lbound < l && board[player+l]) l -= step;
	while (rbound > r && board[player+r]) r += step;
	if (rbound/SHAPE-lbound/SHAPE > 5){
		// y coordinate of bottom(r) minus y coordinate of top(l)
		// vertical distance between r and l
		reward = r/SHAPE-l/SHAPE-1;
		if (lbound != l)
			hboard.at(player+l) += std::pow(4, reward);
		if (rbound != r)
			hboard.at(player+r) += std::pow(4, reward);
	}
	if (r-action-step > 0 && rbound != r)
		hboard.at(player+r) += std::pow(4, r/SHAPE - action/SHAPE-1);
	if (action-l-step > 0 && lbound != l)
		hboard.at(player+l) += std::pow(4, action/SHAPE - l/SHAPE-1);

	/* lr diagonal / */
	l = action-SHAPE+1;
	r = action+SHAPE-1;
	lbound = l;
	rbound = r;
	step = SHAPE-1;

	tmp = x > y ? action-step*(y+1) : action-step*(x+1);
	while (tmp < lbound && !board[op+lbound]) lbound -= step;
	tmp = x > y ? action+(SHAPE-x)*step : action+(SHAPE-y)*step;
	while (tmp > rbound && !board[op+rbound]) rbound += step;

	while (lbound < l && board[player+l]) l -= step;
	while (rbound > r && board[player+r]) r += step;
	if (rbound/SHAPE-lbound/SHAPE > 5){
		// y coordinate of bottom(r) minus y coordinate of top(l)
		// vertical distance between r and l
		reward = r/SHAPE-l/SHAPE-1;
		if (lbound != l)
			hboard.at(player+l) += std::pow(4, reward);
		if (rbound != r)
			hboard.at(player+r) += std::pow(4, reward);
	}
	if (r-action-step > 0 && rbound != r)
		hboard.at(player+r) += std::pow(4, r/SHAPE - action/SHAPE-1);
	if (action-l-step > 0 && lbound != l)
		hboard.at(player+l) += std::pow(4, action/SHAPE - l/SHAPE-1);

	return;
}

void
Cmcts::print_heur()
{
	for (int i = 0; i < SHAPE; i++){
		for (int j = 0; j < SHAPE; j++){
			std::cout << hboard[SIZE+i*SHAPE+j]+hboard[i*SHAPE+j] << " ";
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
	s.append("First player: x\n");
	if (player == 0)
		s.append("Player: x\n");
	else
		s.append("Player: o\n");
	if (get_winner() == 0)
		s.append("Winner: x\n");
	else if (get_winner() == 1)
		s.append("Winner: o\n");
	else if (get_winner() == -1)
		s.append("Winner: _\n");
	else
		s.append("Winner: draw\n");

	s.append("Alpha: "+std::to_string(alpha[0])+"\n");
	if (predict == nullptr)
		s.append("Heuristic: yes\n");
	else
		s.append("Heuristic: no\n");

	s.append("Move count: "+std::to_string(move_cnt)+"\n");
	s.append("Board size: "+std::to_string(SIZE)+"\n");
	s.append("Board:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++)
			if (board[i*SHAPE+j] == 1)
				s.append("x ");
			else if (board[i*SHAPE+j+SIZE] == 1)
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
		if (n->is_null(i)){
			py::print("child is nullptr");
			return;
		}
		n = n->next_node(i);
		std::cout << i << std::endl;
	}
	py::print(n->repr());
	return;
}
