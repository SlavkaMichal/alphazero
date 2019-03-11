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
	py::class_<Cmcts>(m, "mcts")
	.def(py::init<int, double, double>(), "seed"_a=0, "alpha"_a=1, "cpuct"_a=1)
	.def("set_predictor", &Cmcts::set_predictor, "Passes to MCTS funtion to predict prior probability")
	.def("set_seed", &Cmcts::set_seed, "Sets seed for rundom number generator")
	.def("set_alpha", &Cmcts::set_alpha, "Sets alpha variable for generating dirichlet noise")
	.def("set_cpuct", &Cmcts::set_cpuct, "Sets cpuct used to compute UCB")
	.def("clear", &Cmcts::clear, "Clears object state")
	.def("clear_predictor", &Cmcts::clear_predictor, "Clears predictor")
	.def("simulate", &Cmcts::simulate, "Run n MCTS simulation")
	.def("make_move", py::overload_cast<int,int>(&Cmcts::make_move), "y"_a, "x"_a, "Make move")
	.def("make_move", py::overload_cast<int>(&Cmcts::make_move), "Make move")
	.def_property_readonly("player", &Cmcts::get_player, "Player to move")
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
