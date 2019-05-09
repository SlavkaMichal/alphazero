#include "cmcts.h"
#include <ctime>

namespace py = pybind11;

Cmcts::Cmcts(double alpha, double cpuct) :
	cpuct(cpuct),
	cuda(0),
	dir_eps(0.25),
	threads(0)
{
	root_node = new Node();
	state     = new State();
	srand(time(NULL));

	const gsl_rng_type *T;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

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
Cmcts::clear_params()
{
	param_name.clear();
	return;
}

/* getters and setters */
const int Cmcts::get_player() const { return state->player; }
void      Cmcts::set_player(int player) { state->player = player; }

const double Cmcts::get_eps() const { return dir_eps; }
void         Cmcts::set_eps(double eps) {
	if (eps > 1. || eps < 0.)
		throw std::runtime_error("Epsilon must be in range <0,1>");
	dir_eps = eps;
}

const float Cmcts::get_cpuct() const { return this->cpuct; }
void        Cmcts::set_cpuct(float cpuct) { this->cpuct = cpuct; }

const int Cmcts::get_threads() const { return threads; }
void      Cmcts::set_threads(int threads) { this->threads = threads; }

const double Cmcts::get_alpha() const       { return this->alpha[0]; }
void         Cmcts::set_alpha(double alpha) { std::fill(this->alpha, this->alpha+SIZE, alpha); }

const std::string Cmcts::get_params() const                 { return param_name; }
void              Cmcts::set_params(std::string &file_name) { this->param_name = file_name; }

const int Cmcts::get_cuda() const { return cuda; }
void      Cmcts::set_cuda(int cuda) { if (cuda != 1 || cuda != 0) throw std::runtime_error("Invalid value"); this->cuda = cuda; }

const int Cmcts::get_move_cnt() const { return state->move_cnt; }
const float Cmcts::get_winner() const { return state->winner; }

void
Cmcts::set_alpha_default()
{
	double len = SHAPE *2; // average game lenght estimate
	double num = (SIZE*len - (len*len+len)*0.5)/len;
	std::fill(alpha, alpha+SIZE, 10/num);
}

void
Cmcts::simulate(int n)
{
	if (n < 1)
		throw std::runtime_error("Invalid input "+std::to_string(n)+" must be at least 1!");
#ifndef HEUR
	if (param_name.empty()){
		throw std::runtime_error("Predictor missing!");
	}
#endif

	// divide workload
	py::gil_scoped_release release;
	if (threads > 1){
		int th_num = threads;
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

		delete[] threads;
	}
	else {
		worker(n);
	}

	return;
}

void
Cmcts::worker(int n)
{
	torch::NoGradGuard guard;
	std::shared_ptr<torch::jit::script::Module> module = nullptr;
	if (!param_name.empty()){
		module = torch::jit::load(param_name.c_str());
		assert(module != nullptr);
	}
	else{
#ifndef HEUR
		throw std::runtime_error("No module name set");
#endif
	}
	State *search_state = new State(state);

	if (this->cuda)
		module->to(at::kCUDA);

	for (int i = 0; i < n; i++){
		search(search_state, module);
		search_state->clear(state);
	}

	delete search_state;
	return;
}

void
Cmcts::search(State *state, std::shared_ptr<torch::jit::script::Module> module)
{
	Node* current = root_node;
	std::stack<Node*> nodes;
	std::stack<int>   actions;
	float value = 0.;
	int action;
	std::vector<torch::jit::IValue> input;
	std::vector<int64_t> sizes = {1, 2, SHAPE, SHAPE};
	auto options = torch::TensorOptions().dtype(torch::kChar);
	std::vector<torch::jit::IValue> output;
	/* TODO
	   is_end vrati -1 ak player prehral 1 ak player vyhral
	   0 ak hra pokracuje
	   */
	/* Stats
	int end_state = 0;
	int loops = 0;
	int expansion = 0;
	*/
	while (1){
		if (current->nodeN == -1){
			/* node expansion */

			/* generating dirichlet noise */
			gsl_ran_dirichlet(r, SIZE, alpha, dir_noise);
#ifdef HEUR
			/* with heuristic */
			if (module == nullptr){
				value = -rollout();
				current->set_prior(state, dir_noise, dir_eps);
				break;
			}
#endif
			/* no heuristic */
			/* create tensor wraper around buffer */
			at::Tensor tensor;
			tensor = torch::from_blob(
					(void *)state->board.data(),
					at::IntList(sizes),
					options);
			tensor = tensor.toType(at::kFloat); //this will make a copy

			if (this->cuda)
				tensor.to(at::kCUDA);

			input.push_back(tensor);
			try {
				/* evaluate model */
				output = module->forward(input).toTuple()->elements();
			}
			catch (const std::exception& e) {
				// this executes if f() throws std::underflow_error (base class rule)
				std::cout << "Forward failed in thread: " << std::this_thread::get_id() << std::endl;
				std::cout << e.what() << std::endl;
				std::cout << state->repr() << std::endl;
				std::cout << current->repr() << std::endl;
				return;
			}

			value = -output[0].toTensor().to(at::kCPU).item<float>();
			current->set_prior(output[1].toTensor().to(at::kCPU), dir_noise, dir_eps);
			break;
		}

		/* select new node, if it's leaf then expand */
		/* select is also updating node */
		action = current->select(state, cpuct);
		state->make_move(action);
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
			// end_state += 1;
			if (state->winner == 0 || state->winner == 1)
				value = 1.;
			else
				/* result is a draw (winner value is 0.5) */
				value = 0.;
			break;
		}
		current = current->next_node(action);
	}
	/* backpropagate value */
	while (!nodes.empty()){
		current = nodes.top();
		action  = actions.top();
		nodes.pop();
		actions.pop();

		current->backpropagate(action, value);
		value = -value;
	}

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
Cmcts::make_movexy(int x, int y)
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
Cmcts::get_prob()
{
	/* all nonvalid moves should not be played,
	   they are skiped in select phase so visit count should be zero
	 */
	std::array<long int, SIZE>* counts = root_node->counts();
	double sum = root_node->nodeN;
	double prob_sum = 0;
	//auto v = new std::vector<float>(counts->begin(), counts->end());
	auto b = py::array_t<float>(counts->size());
	py::buffer_info buff = b.request();
	if (buff.shape[0] != counts->size())
		throw std::runtime_error("cmcts get_prob size mismatch");
	float *ptr = (float*)buff.ptr;
	for (int i = 0; i < counts->size(); i++){
		ptr[i] = (double)counts->at(i)/sum;
		prob_sum += ptr[i];
	}
	if (round(prob_sum*1000.) != 1000){
		throw std::runtime_error("Not a distribution");
	}
	return b;
}

#ifdef HEUR
py::array_t<float>
Cmcts::get_heur()
{
	auto b = py::array_t<float>(state->hboard.size());
	py::buffer_info buff = b.request();
	if (buff.shape[0] != state->hboard.size())
		throw std::runtime_error("cmcts get_heur size mismatch");
	float *ptr = (float*)buff.ptr;
	for (int i = 0; i < state->hboard.size(); i++)
		ptr[i] = state->hboard[i];
	return b;
	//auto h = new std::vector<float>(state->hboard.begin(),state->hboard.end());
	//auto capsule = py::capsule(h, [](void *h) { delete reinterpret_cast<std::vector<float>*>(h);});
	//return py::array_t<float>(std::vector<ptrdiff_t>{2,SHAPE,SHAPE}, h->data(), capsule);
}

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
	if (param_name.empty())
		s.append("Heuristic: yes\n");
	else{
		s.append("Heuristic: no\n");
		s.append("Parameters from: "+param_name+"\n");
	}

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
	}
	py::print(n->print_u(state, cpuct));
	return;
}
