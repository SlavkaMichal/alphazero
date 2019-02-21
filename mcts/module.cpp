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
	.def(py::init<>())
	.def("add_predictor", &Cmcts::add_predictor)
	.def("clear", &Cmcts::clear)
	.def("simulate", &Cmcts::simulate)
	.def("make_move", &Cmcts::make_move)
	.def("get_result", &Cmcts::get_result)
//	.def("__repr__", &Cmcts::repr)
	.def("get_nn_representation",  &Cmcts::nn_input, py::return_value_policy::take_ownership)
	.def("get_probabilities", &Cmcts::get_prob, py::return_value_policy::take_ownership);
};
