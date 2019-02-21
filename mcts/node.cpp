#include "node.h"
#include <cmath>
#include <math.h>

Node::Node() :
	nodeN(-1),
	edgeN(),
	edgeW(),
	edgeP(),
	children(),
	last_action(-1)
{}

Node::~Node()
{
	for (int i = 0; i < SIZE; i++)
		delete children[i];
	return;
}

void
Node::set_prior(py::array_t<float> p, double *dir)
{
	auto buff = p.request();
	float *ptr = (float *)buff.ptr;
	// copy result
	// TODO som si isty ze toto ide aj lepsie
	for (int i = 0; i < SIZE; i++)
		edgeP[i] = ptr[i]+dir[i];

	nodeN = 0;
	return;
}

void
Node::backpropagate(float value)
{
	if (last_action == -1)
		throw std::runtime_error("No action chosen from this node");
	edgeN[last_action] += 1;
	edgeW[last_action] += value;
	nodeN += 1;

	/* to ensure that a node won't be backpropageted twice with the same action */
	last_action = -1;

	return;
}

int
Node::select(Board &board)
{
	if (nodeN == -1)
		throw std::runtime_error("Node has not been visited yet. Can't select next_node");
	int best_a = -1;
	double u;
	double best_u = -INFINITY;
	for (int a = 0; a < SIZE; a++){
		/* TODO
		   uplne mi to nesedi
		   toto bude vzdy 0 pri prvej navsteve uzlu, co je hovadina
		   v implementaciach som to nevidel nijak riesene iba v jednej pridali eps
		*/
		/* skip non_valid moves */
		// TODO napisat ci pre toto test
		if (board[a] != 0 or board[SIZE+a] != 0)
			continue;
		u = CPUCT * edgeP[a] * std::sqrt(nodeN + 1e-8) / (edgeN[a] + 1);
		if (edgeN[a] != 0)
			u = u + edgeW[a]/edgeN[a];
		if (u > best_u){
			best_u = u;
			best_a = a;
		}
	}

	if (best_a == -1)
		throw std::runtime_error("No action chosen. Incorrect behaviour");

	return best_a;
	//act = best_a;

	//if (children[best_a] == nullptr)
	//	children[best_a] = new Node();
	//return children[best_a];
}

struct Node*
Node::next_node(int action)
{
	Node *ret = nullptr;
	last_action = action;

	if (children[action] == nullptr){
		ret = new Node();
	}
	else{
		/* save pointer to a new node */
		ret = children[action];
		/* remove reference to the node so it won't be deleted with other chil nodes */
		children[action] = nullptr;
	}

	return ret;
}

std::array<int, SIZE>*
Node::counts()
{
	return &edgeN;
}
