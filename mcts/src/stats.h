#ifndef __STATS_H__
#define __STATS_H__

#include <mutex>

struct Stats{
public:
	Stats();
	~Stats();

	void simulations(int sims);
	void loop_stats(int loops, int expansions, int end_state);
	void game();

	void clear();
	std::string repr();

private:
	std::mutex mutex;
	unsigned int games;
	unsigned int sims;
	unsigned int sim_expansions;
	unsigned int sim_end;
	unsigned int sim_loops;
};

#endif
