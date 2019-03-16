#include "cmcts.h"
#include <pybind11/embed.h>

using namespace pybind11::literals;
namespace py = pybind11;

int main()
{
	py::scoped_interpreter guard{};

	py::module init = py::module::import("init");

	Cmcts mcts = Cmcts(1, 0.5, init.attr("CPUCT").cast<double>());
	py::object obj = init.attr("model_wraper");
	py::object model = init.attr("model");
	std::function<py::tuple(py::array_t<float>, py::object)> predictor = obj;
	mcts.set_predictor(predictor,model);
	mcts.simulate(1);

	return 0;
}
