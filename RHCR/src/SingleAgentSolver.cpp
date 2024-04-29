#include "SingleAgentSolver.h"

double SingleAgentSolver::compute_h_value(
    const BasicGraph &G, int curr, int goal_id,
    const vector<tuple<int, int, int>> &goal_location) const
{
    double h = G.heuristics.at(std::get<0>(goal_location[goal_id]))[curr];
    goal_id++;
    while (goal_id < (int)goal_location.size())
    {
        h += G.heuristics.at(
                std::get<0>(goal_location[goal_id])
            )[std::get<0>(goal_location[goal_id - 1])];
        goal_id++;
    }
    return h;
}
