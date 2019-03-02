#include "node.h"
#include "cmcts.h"

using namespace pybind11::literals;

int test(int i)
{
	return i+1;
}

PYBIND11_MODULE(cmcts, m) {
	m.doc() = "This is documentation for cmts module";
	m.def("test", &test, "Documentation for function");
	py::class_<Cmcts>(m, "mcts")
	.def(py::init<int>(), "seed"_a=0)
	.def("set_predictor", &Cmcts::set_predictor, "This function passes to MCTS funtion to predict prior probability")
	.def("clear", &Cmcts::clear, "Clear object state")
	.def("simulate", &Cmcts::simulate, "Run n MCTS simulation")
	.def("make_move", py::overload_cast<int,int>(&Cmcts::make_move), "y"_a, "x"_a, "Make move")
	.def("make_move", py::overload_cast<int>(&Cmcts::make_move), "Make move")
	.def_property_readonly("player", &Cmcts::get_player, "Player to move")
	.def_property_readonly("winner", &Cmcts::get_winner, "Player to move")
	.def_property_readonly("move_cnt", &Cmcts::get_move_cnt, "Number of played moves")
	.def("is_end", &Cmcts::is_end, "Returns 0 if there are valid moves\n\t1 if player to move wins\n\t-1 if player to move lost")
	.def("print_node", &Cmcts::print_node)
#ifdef HEUR
	.def("print_heur", &Cmcts::print_heur)
	.def("heur",  &Cmcts::get_heur, py::return_value_policy::take_ownership, "Current hboard")
#endif
	.def("board",  &Cmcts::get_board, py::return_value_policy::take_ownership, "Board representation")
	.def("prob", &Cmcts::get_prob, py::return_value_policy::take_ownership, "Returns probabilities for each move leading to win")
	.def("__repr__", &Cmcts::repr);
};
