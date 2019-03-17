#include "cmcts.h"
#include <pybind11/embed.h>

using namespace pybind11::literals;
namespace py = pybind11;

int main()
{
	py::scoped_interpreter guard{};

	Cmcts mcts = Cmcts(1, 0.5, 5);
	std::string file ="/home/michal/workspace/bp/alphazero/src/simplerNN13.pt";
	mcts.set_params(file);
	mcts.simulate(100);
	mcts.make_move(0);
	mcts.simulate(1);
	std::cout << mcts.repr();

	return 0;
}
