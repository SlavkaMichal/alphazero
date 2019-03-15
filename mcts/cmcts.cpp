#include "cmcts.h"

namespace py = pybind11;

Cmcts::Cmcts(uint64_t seed, double alpha, double cpuct) :
	cpuct(cpuct)
{
	root_node = new Node();
	state     = new State();

	const gsl_rng_type *T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r, seed);

	this->alpha = new double[SIZE];
	dir_noise   = new double[SIZE];
	// x = avg_game_length = SHAPE*2
	// 10/((SIZE*x-(x**2+x)*0.5)/x)
	std::fill(this->alpha, this->alpha+SIZE, alpha);
}

Cmcts::~Cmcts(void)
{
	delete root_node;
	delete state;
	gsl_rng_free(r);
	delete[] alpha;
	delete[] dir_noise;
}

void
Cmcts::clear()
{
	delete root_node;
	root_node = new Node();

	state->clear();
	return;
}

void
Cmcts::clear_predictor()
{
	predict = nullptr;
	return;
}

int
Cmcts::get_player()
{
	return state->player;
}

int
Cmcts::get_move_cnt()
{
	return state->move_cnt;
}

float
Cmcts::get_winner()
{
	return state->winner;
}

void
Cmcts::set_cpuct(float cpuct)
{
	this->cpuct = cpuct;
	return;
}
void
Cmcts::set_alpha(double alpha)
{
	std::fill(this->alpha, this->alpha+SIZE, alpha);
	return;
}

void
Cmcts::set_predictor(std::function<py::tuple(py::array_t<float>, py::object)> &p, py::object &d)
{
	predict = p;
	data    = d;

	double len = SHAPE *2; // average game lenght estimate
	double num = (SIZE*len - (len*len+len)*0.5)/len;
	std::fill(alpha, alpha+SIZE, 10/num);
	return;
}

void
Cmcts::set_seed(unsigned long int seed)
{
	gsl_rng_set(r, time(NULL)+seed);
	return;
}

void
Cmcts::simulate(int n)
{
	if (n < 1)
		throw std::runtime_error("Invalid input "+std::to_string(n)+" must be at least 1!");
#ifndef HEUR
	if (predict == nullptr){
		throw std::runtime_error("Predictor missing!");
	}
#endif

	// divide workload
	py::gil_scoped_release release;
#ifdef THREADS
	int th_num = THREADS;
	if (n < th_num){
		th_num = n;
		n = 1;
	}
	else if (n % th_num == 0)
		n = n/th_num;
	else
		n = n/th_num + 1;

	std::thread *threads = new std::thread[th_num];

	for (int i = 0; i < th_num; i++){
		threads[i] = std::thread(&Cmcts::worker, this, n);
	}
	for (int i = 0; i< th_num; i++){
		threads[i].join();
	}

	std::cout<< "delete threads" <<std::endl;
	delete[] threads;
	std::cout<< "deleted threads" <<std::endl;
#else
	worker(n);
#endif
	return;
}

void
Cmcts::worker(int n)
{
	std::cout<<std::this_thread::get_id()<< " started" <<std::endl;
	std::cout<<std::this_thread::get_id()<< " has "<<n <<" jobs" <<std::endl;
	State *search_state = new State(state);

	for (int i = 0; i < n; i++){
		std::cout<<std::this_thread::get_id()<< " job "<<i <<std::endl;
		std::cout<<std::this_thread::get_id()<< " "<<search_state <<std::endl;
		search(search_state);
		std::cout<<std::this_thread::get_id()<< " search returned "<<i <<std::endl;
		std::cout<<std::this_thread::get_id()<< " "<<search_state <<std::endl;
		search_state->clear(state);
	}

	std::cout<<std::this_thread::get_id()<< " delete state" <<std::endl;
	delete search_state;
	std::cout<<std::this_thread::get_id()<< " deleted state" <<std::endl;
}

void
Cmcts::search(State *state)
{
	Node* current = root_node;
	std::stack<Node*> nodes;
	std::stack<int>   actions;
	float value = 0.;
	int action;

	py::tuple prediction_t;

	/* TODO
	   is_end vrati -1 ak player prehral 1 ak player vyhral
	   0 ak hra pokracuje
	   */
	while (1){
		if (current->nodeN == -1){
			/* node expansion */
			std::cout<<std::this_thread::get_id()<< " initializing node" <<std::endl;
			gsl_ran_dirichlet(r, SIZE, alpha, dir_noise);
#ifdef HEUR
			std::cout<<std::this_thread::get_id()<< " heur def" <<std::endl;
			if (predict != nullptr){
				std::cout<<std::this_thread::get_id()<< " not null" <<std::endl;
				py::gil_scoped_acquire acquire;
				auto board =state->get_board();
				prediction_t = predict(board, data);
				std::cout<<std::this_thread::get_id()<< " cast" <<std::endl;
				value = -prediction_t[0].cast<float>();
				std::cout<<std::this_thread::get_id()<< " set" <<std::endl;
				current->set_prior(prediction_t[1].cast<py::array_t<float>>(), dir_noise);
			}else{
				value = -rollout();
				current->set_prior(state, dir_noise);
			}
#else
			//std::cout<<"2gil: " <<PyGILState_Check()<< std::endl;
			std::cout<<std::this_thread::get_id()<< " else" <<std::endl;
			py::gil_scoped_acquire acquire;
			prediction_t = predict(state->get_board(), data);
			value = -prediction_t[0].cast<float>();
			current->set_prior(prediction_t[1].cast<py::array_t<float>>(), dir_noise);
#endif
			std::cout <<std::this_thread::get_id()<< " search new node explored" << std::endl;
			break;
		}

		/* select new node, if it's leaf then expand */
		/* select is also updating node */
		std::cout<<std::this_thread::get_id()<< " select" <<std::endl;
		action = current->select(state, cpuct);
		std::cout<<std::this_thread::get_id()<< " make_move" <<std::endl;
		state->make_move(action);
		std::cout<<std::this_thread::get_id()<< " push"<< current <<std::endl;
		nodes.push(current);
		actions.push(action);

		/* returns nullptr if reaches final state */
		if (state->winner != -1){
			/* if game is over don't push leaf node */
			/* so what went wrong?? game is over in the next move,
			   therefor I have to push this node to get updates which is the winning move
			   but I don't have to create new node with end state */
			/* player on move is the one who lost */
			/* last node on stack is previous and that one also lost */
			if (state->winner == 0 || state->winner == 1)
				value = 1.;
			else
				value = 0.;
			break;
		}
		current = current->next_node(action);
	}
	std::cout <<std::this_thread::get_id()<< " ready to backprop" << std::endl;
	/* backpropagate value */
	while (!nodes.empty()){
		std::cout <<std::this_thread::get_id()<< " ready to backprop" << std::endl;
		current = nodes.top();
		std::cout<<std::this_thread::get_id()<< " pop"<< current <<std::endl;
		action  = actions.top();
		nodes.pop();
		actions.pop();

		std::cout<<std::this_thread::get_id()<< " backprop" <<std::endl;
		current->backpropagate(action, value);
		value = -value;
	}
	std::cout <<std::this_thread::get_id()<< " search return" << std::endl;

	return;
}

void
Cmcts::make_move(int action)
{
	Node *old_node = root_node;

	state->make_move(action);
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

	state->make_move(action);
	root_node = root_node->make_move(action);

	delete old_node;
	return;
}

py::array_t<float>
Cmcts::get_board()
{
	return state->get_board();
}

py::array_t<float>
Cmcts::get_heur()
{
	auto b = py::array_t<float>(state->hboard.size());
	py::buffer_info buff = b.request();
	float *ptr = (float*)buff.ptr;
	for (int i = 0; i < state->hboard.size(); i++)
		ptr[i] = state->hboard[i];
	return b;
	//auto h = new std::vector<float>(state->hboard.begin(),state->hboard.end());
	//auto capsule = py::capsule(h, [](void *h) { delete reinterpret_cast<std::vector<float>*>(h);});
	//return py::array_t<float>(std::vector<ptrdiff_t>{2,SHAPE,SHAPE}, h->data(), capsule);
}

py::array_t<float>
Cmcts::get_prob()
{
	/* all nonvalid moves should not be played,
	   they are skiped in select phase so visit count should be zero
	 */
	std::array<int, SIZE>* counts = root_node->counts();
	int sum = root_node->nodeN;
	//auto v = new std::vector<float>(counts->begin(), counts->end());
	auto b = py::array_t<float>(counts->size());
	py::buffer_info buff = b.request();
	float *ptr = (float*)buff.ptr;
	for (int i = 0; i < state->hboard.size(); i++)
		ptr[i] = counts->at(i);
	return b;

	//for (int i = 0; i < SIZE; i++)
	//	v->at(i) = v->at(i) / sum;
	//auto capsule = py::capsule(v, [](void *v) { delete reinterpret_cast<std::vector<float>*>(v);});

	//return py::array_t<float>(v->size(), v->data(), capsule);
}

#ifdef HEUR
float
Cmcts::rollout()
{
	float value = 0;
	State *rollout_state = new State(state);

	while (rollout_state->winner == -1){
		/* choose action */
		rollout_state->rollout_move();
	}

	if (rollout_state->winner == 1. || rollout_state->winner == 0.)
		value = state->player == rollout_state->winner ? 1. : -1.;

	delete rollout_state;
	return value;
}

void
Cmcts::print_heur()
{
	state->print_heur();
	return;
}
#endif

std::string
Cmcts::repr()
{
	std::string s;
	s.append("Alpha: "+std::to_string(alpha[0])+"\n");
	s.append("CPUCT: "+std::to_string(cpuct)+"\n");
	if (predict == nullptr)
		s.append("Heuristic: yes\n");
	else
		s.append("Heuristic: no\n");

	s.append(state->repr());

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

void
Cmcts::print_u(std::vector<int> &v)
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
	py::print(n->print_u(state, cpuct));
	return;
}
