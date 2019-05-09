#include "state.h"

State::State() :
	player(0),
	move_cnt(0),
	winner(-1.)
{
	std::fill(board.begin(), board.end(), 0);
#ifdef HEUR
	std::fill(hboard.begin(), hboard.end(), 0.5);
	hboard.at(hboard.size()/4) += 0.5;
	hboard.at((hboard.size()/4)*3) += 0.5;
#endif
}

State::State(const State *obj)
{
	player   = obj->player;
	move_cnt = obj->move_cnt;
	winner   = obj->winner;
	board    = obj->board;
#ifdef HEUR
	hboard   = obj->hboard;
#endif
}

State::~State() {}

bool
State::is_valid(int action)
{
	if (action >= SIZE || action < 0)
		return false;
	if (board[action] == 1 || board[SIZE+action] == 1)
		return false;
	return true;
}

py::array_t<float>
State::get_board()
{
	auto b = new std::vector<float>(board.begin(),board.end());
	if (player == 1)
		std::swap_ranges(b->begin(), b->begin()+SIZE, b->begin()+SIZE);

	auto capsule = py::capsule(b, [](void *b) { delete reinterpret_cast<std::vector<float>*>(b);});
	return py::array_t<float>(std::vector<ptrdiff_t>{2,SHAPE,SHAPE}, b->data(), capsule);
}

void
State::clear(const State *obj)
{
	player   = obj->player;
	move_cnt = obj->move_cnt;
	winner   = obj->winner;
	board    = obj->board;
	moves    = obj->moves;
#ifdef HEUR
	hboard   = obj->hboard;
#endif
	return;
}

void
State::clear()
{
	player   = 0;
	move_cnt = 0;
	winner   = -1;
	moves.clear();

	std::fill(board.begin(), board.end(), 0);
#ifdef HEUR
	std::fill(hboard.begin(), hboard.end(), 0.5);
	hboard.at(hboard.size()/4) += 0.5;
	hboard.at((hboard.size()/4)*3) += 0.5;
#endif
	return;
}

void
State::make_move(int action)
{
	float end = 0;
	if (!is_valid(action))
		throw std::runtime_error("Invalid move "+std::to_string(action)+"\n "+repr()+"!");

	board.at(player*SIZE + action) = 1;
	player = player ? 0 : 1;
	move_cnt += 1;
	moves.push_back(action);

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

std::string
State::repr()
{
	std::string s;
	s.append("First player: x\n");
	if (player == 0)
		s.append("Player: x\n");
	else
		s.append("Player: o\n");
	if (winner == 0)
		s.append("Winner: x\n");
	else if (winner == 1)
		s.append("Winner: o\n");
	else if (winner == -1)
		s.append("Winner: _\n");
	else
		s.append("Winner: draw\n");

	s.append("Move count: "+std::to_string(move_cnt)+"\n");
	s.append("Moves: ");
	for (int i = 0; i < moves.size(); ++i){
		s.append(std::to_string(moves[i])+", ");
	}
	s.append("\n");
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

float
State::is_end()
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
void
State::rollout_move()
{
	int action  = 0;

	if (this->winner != -1.)
		return;
	while(1){
		action = rand()%SIZE;
		if (is_valid(action))
			break;
	}

	make_move(action);
}

void
State::print_heur()
{
	for (int i = 0; i < SHAPE; i++){
		for (int j = 0; j < SHAPE; j++){
			if (hboard[SIZE+i*SHAPE+j]+hboard[i*SHAPE+j] == 1)
				std::cout << "_ ";
			else
				std::cout << hboard[SIZE+i*SHAPE+j]+hboard[i*SHAPE+j] << " ";
		}
		std::cout << std::endl;
	}
	return;
}

void
State::update(int action)
{
	int l = 0, r = 0;
	int lbound = 0, rbound = 0;
	int x = 0, y = 0, tmp = 0;
	int op = this->player == 1 ? SIZE : 0;
	int player = this->player == 1 ? 0 : SIZE;
	int step = 0;
	int reward = 0;

	y = action/SHAPE;
	x = action%SHAPE;
	hboard.at(SIZE+action) = 0;
	hboard.at(action) = 0;

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

	reward = r-l-1;
	if (reward == 4)
		reward *= 2;
	for (int i = 4; i >= 0; i--){
		if (lbound >= l)
			break;
		if (rbound-lbound > 5){
			hboard.at(player+l) += reward*i;
		}
		l -= step;
	}
	for (int i = 4; i >= 0; i--){
		if (rbound <= r)
			break;
		if (rbound-lbound > 5){
			//hboard.at(player+r) += std::pow(reward,i);
			hboard.at(player+r) += reward*i;
		}
//		if (r-action-step > i*step){
//			hboard.at(player+r) -= std::pow(r-action-1,3);
//		}
		r += step;
	}


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

	reward = (r-l)/step-1;
	if (reward == 4)
		reward *= 2;
	for (int i = 4; i >= 0; i--){
		if (lbound >= l)
			break;
		if (rbound-lbound > 5*SHAPE){
			hboard.at(player+l) += reward*i;
		}
		l -= step;
	}
	for (int i = 4; i >= 0; i--){
		if (rbound <= r)
			break;
		if (rbound-lbound > 5*SHAPE){
			hboard.at(player+r) += reward*i;
		}
		r += step;
	}

	/* lr diagonal \ */
	l = action-SHAPE-1;
	r = action+SHAPE+1;
	lbound = l;
	rbound = r;
	step = SHAPE+1;

	tmp = x-y >= 0 ? x-y-step : -1 + (y-x-1)*SHAPE;
	while (tmp < lbound && !board[op+lbound]) lbound -= step;
	tmp = x > y ? action+(SHAPE-x)*step : action+(SHAPE-y)*step;
	tmp = x-y >= 0 ? step*SHAPE -SHAPE*(x-y) : step*SHAPE + x - y;
	while (tmp > rbound && !board[op+rbound]) rbound += step;

	while (lbound < l && board[player+l]) l -= step;
	while (rbound > r && board[player+r]) r += step;

	reward = (r-l)/step-1;
	if (reward == 4)
		reward *= 2;
	for (int i = 4; i >= 0; i--){
		if (lbound >= l)
			break;
		if ((rbound-lbound)/step > 5){
			hboard.at(player+l) += reward*i;
			//hboard.at(player+l) += std::pow(reward,i);
		}
		l -= step;
	}
	for (int i = 4; i >= 0; i--){
		if (rbound <= r)
			break;
		if ((rbound-lbound)/step > 5){
			hboard.at(player+r) += reward*i;
		}
//		if (r-action-step > i*step){
//			hboard.at(player+r) -= std::pow(r/SHAPE-action/SHAPE-1,3);
//		}
		r += step;
	}

	/* lr diagonal / */
	l = action-SHAPE+1;
	r = action+SHAPE-1;
	lbound = l;
	rbound = r;
	step = SHAPE-1;

	tmp = x+y < SHAPE-1 ? x+y-step : (y+x-step)*SHAPE;
	while (tmp < lbound && !board[op+lbound]) lbound -= step;
	tmp = x+y < SHAPE-1 ? (x+y)*SHAPE+step : step*SHAPE + x + y;
	while (tmp > rbound && !board[op+rbound]) rbound += step;

	while (lbound < l && board[player+l]) l -= step;
	while (rbound > r && board[player+r]) r += step;

	reward = (r-l)/step-1;
	if (reward == 4)
		reward *= 2;
	for (int i = 4; i >= 0; i--){
		if (lbound >= l)
			break;
		if ((rbound-lbound)/step > 5){
			hboard.at(player+l) += reward*i;
		}
		l -= step;
	}
	for (int i = 4; i >= 0; i--){
		if (rbound <= r)
			break;
		if ((rbound-lbound)/step > 5){
			hboard.at(player+r) += reward*i;
		}
		r += step;
	}

	return;
}
#endif
