#include "node.h"
#include "cmcts.h"

using namespace pybind11::literals;

int test(int i)
{
	std::cout << MAJOR << "." << MINOR << std::endl;
	return i+1;
}

PYBIND11_MODULE(cmcts, m) {
	m.doc() = "This is documentation for CMTS module";
	m.def("test", &test, "Documentation for function");
	m.def("version", []() { return GIT; } );
	m.def("build_timestamp", []() { std::string s; return s+__DATE__+" "+__TIME__; } );
	py::class_<Cmcts>(m, "mcts")
	.def(py::init<int, double, double>(), "seed"_a=0, "alpha"_a=1, "cpuct"_a=1)
	.def_property("params",
			&Cmcts::get_params,
			&Cmcts::set_params,
			"Passes to MCTS funtion to predict prior probability")
	.def_property("eps",
			&Cmcts::get_eps,
			&Cmcts::set_eps,
			"How significant dirichlet noise should be")
	.def_property("cuda",
			&Cmcts::get_cuda,
			&Cmcts::set_cuda,
			"Evaluate network on graphics card")
	.def_property("alpha",
			&Cmcts::get_alpha,
			&Cmcts::set_alpha,
			"Sets alpha variable for generating dirichlet noise")
	.def_property("threads",
			&Cmcts::get_threads,
			&Cmcts::set_threads,
			"Number of threads mcts runs in")
	.def_property("player",
			&Cmcts::get_player,
			&Cmcts::set_player,
			"Player to move")
	.def_property("cpuct",
			&Cmcts::get_cpuct,
			&Cmcts::set_cpuct,
			"Sets cpuct used to compute UCB")
	.def("set_params", &Cmcts::set_params, "Passes to MCTS funtion to predict prior probability")
	.def("set_seed", &Cmcts::set_seed, "Sets seed for rundom number generator")
	.def("set_alpha", &Cmcts::set_alpha, "Sets alpha variable for generating dirichlet noise")
	.def("set_threads", &Cmcts::set_threads, "Number of threads mcts runs in")
	.def("set_alpha_default", &Cmcts::set_alpha_default, "Sets alpha variable for generating dirichlet noise")
	.def("set_cpuct", &Cmcts::set_cpuct, "Sets cpuct used to compute UCB")
	.def("clear", &Cmcts::clear, "Clears object state")
	.def("clear_params", &Cmcts::clear_params, "Clears predictor")
	.def("simulate", &Cmcts::simulate, "Run n MCTS simulation")
	.def("make_movexy", &Cmcts::make_movexy, "x"_a, "y"_a, "Make move by passing coordinates")
	.def("make_move", &Cmcts::make_move, "Make move")
	.def_property_readonly("winner", &Cmcts::get_winner, "Winner of the game")
	.def_property_readonly("move_cnt", &Cmcts::get_move_cnt, "Number of played moves")
	.def("print_node", &Cmcts::print_node)
	.def("print_u", &Cmcts::print_u)
#ifdef HEUR
	.def("print_heur", &Cmcts::print_heur)
	.def("heur",  &Cmcts::get_heur, py::return_value_policy::take_ownership, "Current hboard")
#endif
	.def("get_board",  &Cmcts::get_board, py::return_value_policy::take_ownership, "Board representation")
	.def("get_prob", &Cmcts::get_prob, py::return_value_policy::take_ownership, "Returns probabilities for each move leading to win")
	.def("__repr__", &Cmcts::repr);
};
