#include "node.h"

Node::
//Node(std::string &name, int a) :
Node() :
	nodeN(-1),
	child{},
	child_cnt(0),
	childN(),
	childP(),
	childW()
{
//	if (a == -1)
//		this->name = name;
//	else
//		this->name = name+":"+std::to_string(a);
//	//std::cout << "Creating node " << this->name <<std::endl;
}

Node::
~Node()
{
	//std::cout << "Deleting node " << this->name <<std::endl;
}

void Node::
set_prior(torch::Tensor p, gsl_rng *r, const double *alpha, double dir_eps)
{
	std::lock_guard<std::mutex> lock(node_mutex);
	if (nodeN != -1){
		// some other thread did the job
		return;
	}
	float *ptr = (float *)p.data_ptr();
	// copy result
	// TODO som si isty ze toto ide aj lepsie

	gsl_ran_dirichlet(r, childP.size(), alpha, childP.data());
	double prior_eps = 1. - dir_eps;
	for (int i = 0; i < childP.size(); i++){
		// dir sum to 1 also p should
		childP[i] = prior_eps*exp(ptr[i]) + dir_eps*childP[i];
	}

	nodeN = 0;
	return;
}

#ifdef HEUR
void Node::
set_prior(State *state, gsl_rng *r, const double *alpha, const double dir_eps)
{
	float sum = 0.;
	std::lock_guard<std::mutex> lock(node_mutex);
	// check if node wasn't already explored
	if (nodeN != -1){
		return;
	}

	gsl_ran_dirichlet(r, childP.size(), alpha, childP.data());
	for (int i = 0; i < childP.size(); i++){
		childP[i] = state->hboard[SIZE+i]+state->hboard[SIZE+i];
		sum += childP[i];
	}
	double prior_eps = 1. - dir_eps;
	for (int i = 0; i < childP.size(); i++){
		childP[i] = childP[i]/sum*prior_eps + childP[i]*dir_eps;
	}

	nodeN = 0;
	return;
}
#endif

void Node::
backpropagate(int action, float value)
{
	std::lock_guard<std::mutex> lock(node_mutex);
	/* restore virtual loss and result of the game*/
	childW.at(action) += value + 1.;

	return;
}

int Node::
select(State *state, double cpuct)
{
	if (nodeN == -1)
		throw std::runtime_error("Node has not been visited yet. Can't select next_node");
	std::lock_guard<std::mutex> lock(node_mutex);
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
		if (!state->is_valid(a))
			continue;
		if (childN[a] != 0)
			u = childW[a]/childN[a] + cpuct*childP[a]*std::sqrt(nodeN + 1e-8) / (childN[a] + 1);
		else
			u = cpuct*childP[a]*std::sqrt(nodeN + 1e-8);

		if (u > best_u){
			best_u = u;
			best_a = a;
		}
	}
	childN.at(best_a) += 1;
	nodeN += 1;
	// subtract virtual loss
	childW.at(best_a) -= 1.;

	if (best_a == -1)
		throw std::runtime_error("No action chosen. Incorrect behaviour");

	return best_a;
}

struct Node* Node::
next_node(int action)
{
	if (child[action]==nullptr){
		child.at(action) = std::unique_ptr<Node>(new Node());
		//child[action] = std::unique_ptr<Node>(new Node(name,action));
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
		//ret = new Node(name, action);
		ret = new Node();
	}

	return ret;
}

std::array<long int, SIZE>* Node::
counts()
{
	return &childN;
}

std::string Node::
repr()
{
	std::string s;
	int sum = 0;
	double sumf = 0;
	std::stringstream ss;

	s.append("Visits: "+std::to_string(nodeN)+"\n");
	//s.append("Child count: "+std::to_string(child_cnt)+"\n");
	//s.append("Name: "+name+"\n");
	s.append("\nCounts:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++){
			ss << std::setw(3) << std::setfill(' ') << childN[i*SHAPE+j];
			s.append(ss.str()+" ");
			ss.str(std::string());
			sum += childN[i*SHAPE+j];
		}
		s.append("\n");
	}
	s.append("\nCounts total: "+std::to_string(sum)+"\n");

	sumf = 0.;
	s.append("\nProbs:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++){
			ss << std::setw(3)<<std::setprecision(3) << std::setfill(' ') << childP[i*SHAPE+j];
			s.append(ss.str()+" ");
			ss.str(std::string());
			sumf += childP[i*SHAPE+j];
		}
		s.append("\n");
	}
	s.append("\nProbs total: "+std::to_string(sumf)+"\n");

	sumf = 0.;
	s.append("\nTotal child values:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++){
			ss << std::setw(3)<<std::setprecision(3) << std::setfill(' ') << childW[i*SHAPE+j];
			s.append(ss.str()+" ");
			ss.str(std::string());
			sumf += childW[i*SHAPE+j];
		}
		s.append("\n");
	}
	s.append("\nSum of total child values: "+std::to_string(sumf)+"\n");

	return s;
}

std::string Node::
print_u(State *state, double cpuct)
{
	std::string s;
	std::stringstream ss;
	if (nodeN == -1)
		throw std::runtime_error("Node has not been visited yet. Can't select next_node");
	int best_a = -1;
	double u;
	double best_u = -INFINITY;
	//s.append("Name: "+name+"\n");
	s.append("\nTotal child values:\n");
	for (int i=0; i<SHAPE; i++){
		for (int j=0; j<SHAPE; j++){
			if (!state->is_valid(i*SHAPE+j)){
				s.append(" ___ ");
				continue;
			}
			if (childN[i*SHAPE+j] != 0)
				u = childW[i*SHAPE+j]/childN[i*SHAPE+j] + cpuct*childP[i*SHAPE+j]*std::sqrt(nodeN + 1e-8) / (childN[i*SHAPE+j] + 1);
			else
				u = cpuct*childP[i*SHAPE+j]*std::sqrt(nodeN + 1e-8);

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

	return s;
}
