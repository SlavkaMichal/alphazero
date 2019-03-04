#include "node.h"
#include <cmath>
#include <math.h>

Node::
Node(std::string &name, int a) :
	nodeN(-1),
	child{},
	last_action(-1),
	child_cnt(0),
	edgeN(),
	edgeP(),
	edgeW()
{
	if (a == -1)
		this->name = name;
	else
		this->name = name+":"+std::to_string(a);
	//std::cout << "Creating node " << this->name <<std::endl;
}

Node::
~Node()
{
	//std::cout << "Deleting node " << this->name <<std::endl;
}

void Node::
set_prior(py::array_t<float> p, double *dir)
{
	auto buff = p.request();
	float *ptr = (float *)buff.ptr;
	// copy result
	// TODO som si isty ze toto ide aj lepsie
	for (int i = 0; i < SIZE; i++){
		// dir sum to 1 also p should
		edgeP[i] = 0.75*ptr[i]+0.25*dir[i];
	}

	nodeN = 0;
	return;
}

void Node::
set_prior(std::array<double, 2*SIZE> &hboard, double* dir)
{
	float sum = 0.;
	for (int i = 0; i < SIZE; i++){
		edgeP[i] = hboard[SIZE+i]+hboard[SIZE+i];
		sum += edgeP[i];
	}
	for (int i = 0; i < SIZE; i++){
		edgeP[i] = edgeP[i]/sum*0.9+dir[i]*0.1;
	}

	nodeN = 0;
	return;
}

void Node::
backpropagate(float value)
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

int Node::
select(Board &board)
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

	last_action = best_a;
	return best_a;
}

struct Node* Node::
next_node(int action)
{
	if (child[action]==nullptr){
		child[action] = std::unique_ptr<Node>(new Node(name,action));
		child_cnt += 1;
	}

	return child[action].get();
}

bool Node::
is_null(int a)
{
	if (a < 0 || a >= SIZE)
		throw std::runtime_error("Index is out of bounds");

	if (child[a]==nullptr)
		return true;
	return false;
}

struct Node* Node::
make_move(int action)
{
	Node *ret = child[action].release();

	if (ret == nullptr){
		ret = new Node(name, action);
		child_cnt += 1;
	}

	return ret;
}

std::array<int, SIZE>* Node::
counts()
{
	return &edgeN;
}

std::string Node::
repr()
{
	std::string s;
	int sum = 0;
	float sumf = 0;
	std::stringstream ss;

	s.append("Visits: "+std::to_string(nodeN)+"\n");
	s.append("Child count: "+std::to_string(child_cnt)+"\n");
	s.append("Last action: "+std::to_string(last_action)+"\n");
	s.append("Name: "+name+"\n");
	s.append("\nCounts:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++){
			ss << std::setw(3) << std::setfill(' ') << edgeN[i*SHAPE+j];
			s.append(ss.str()+" ");
			ss.str(std::string());
			sum += edgeN[i*SHAPE+j];
		}
		s.append("\n");
	}
	s.append("\nCounts total: "+std::to_string(sum)+"\n");

	sum = 0;
	s.append("\nProbs:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++){
			ss << std::setw(3)<<std::setprecision(3) << std::setfill(' ') << edgeP[i*SHAPE+j];
			s.append(ss.str()+" ");
			ss.str(std::string());
			sumf += edgeP[i*SHAPE+j];
		}
		s.append("\n");
	}
	s.append("\nProbs total: "+std::to_string(sumf)+"\n");

	sumf = 0;
	s.append("\nTotal edge values:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++){
			ss << std::setw(3)<<std::setprecision(3) << std::setfill(' ') << edgeW[i*SHAPE+j];
			s.append(ss.str()+" ");
			ss.str(std::string());
			sumf += edgeW[i*SHAPE+j];
		}
		s.append("\n");
	}
	s.append("\nSum of total edge values: "+std::to_string(sumf)+"\n");

	return s;
}

std::string Node::
print_u(Board &board)
{
	std::string s;
	std::stringstream ss;
	if (nodeN == -1)
		throw std::runtime_error("Node has not been visited yet. Can't select next_node");
	int best_a = -1;
	double u;
	double best_u = -INFINITY;
	s.append("Name: "+name+"\n");
	s.append("\nTotal edge values:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++){
			if (board[i*SHAPE+j] == 1 or board[i*SHAPE+j+SIZE] == 1){
				s.append(" ___ ");
				continue;
			}
			if (edgeN[i*SHAPE+j] != 0)
				u = edgeW[i*SHAPE+j]/edgeN[i*SHAPE+j] + CPUCT*edgeP[i*SHAPE+j]*std::sqrt(nodeN + 1e-8) / (edgeN[i*SHAPE+j] + 1);
			else
				u = CPUCT*edgeP[i*SHAPE+j]*std::sqrt(nodeN + 1e-8);

			if (u > best_u){
				best_u = u;
				best_a = i*SHAPE+j;
			}
			ss << std::setw(2) << std::fixed << std::setprecision(3) << std::setfill(' ') << u;
			s.append(ss.str()+" ");
			ss.str(std::string());
		}
		s.append("\n");
	}
	s.append("\nbest action: "+std::to_string(best_a)+"\n");

	if (best_a == -1)
		throw std::runtime_error("No action chosen. Incorrect behaviour");
	s.append("Board:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++)
			if (board[i*SHAPE+j] == 1)
				s.append("x ");
			else if (board[i*SHAPE+j+SIZE] == 1)
				s.append("o ");
			else
				s.append("_ ");
		s.append("\n");
	}

	return s;
}
