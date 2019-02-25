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
	.def("add_predictor", &Cmcts::add_predictor)
	.def("clear", &Cmcts::clear)
	.def("simulate", &Cmcts::simulate)
	.def("make_move", &Cmcts::make_move)
	.def("is_end", &Cmcts::is_end)
	.def("print_node", &Cmcts::print_node)
#ifdef HEUR
	.def("print_heur", &Cmcts::print_heur)
#endif
	.def("__repr__", &Cmcts::repr)
	.def("nn_representation",  &Cmcts::nn_input, py::return_value_policy::take_ownership)
	.def("probabilities", &Cmcts::get_prob, py::return_value_policy::take_ownership);
};
