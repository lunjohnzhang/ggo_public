#include "KivaSystem.h"
#include "WHCAStar.h"
#include "ECBS.h"
#include "LRAStar.h"
#include "PBS.h"
#include "helper.h"
#include "common.h"
#include <algorithm>

KivaSystem::KivaSystem(const KivaGrid &G, MAPFSolver &solver) : BasicSystem(G, solver), G(G) {}

KivaSystem::~KivaSystem()
{
}

void KivaSystem::initialize()
{
	initialize_solvers();

	starts.resize(num_of_drives);
	goal_locations.resize(num_of_drives);
	paths.resize(num_of_drives);
	finished_tasks.resize(num_of_drives);
    waited_time.resize(num_of_drives, 0);
    is_tasking.resize(num_of_drives, false);
	bool succ = load_records(); // continue simulating from the records
	if (!succ)
	{
		timestep = 0;
		succ = load_locations();
		if (!succ)
		{
			cout << "Randomly generating initial start locations" << endl;
			initialize_start_locations();
            cout << "Randomly generating initial goal locations" << endl;
			initialize_goal_locations();
		}
	}

	// Initialize next_goal_type to "w" under w mode
	if (G.get_w_mode())
    {
        for (int i = 0; i < num_of_drives; i++)
        {
            int start_loc = this->starts[i].location;
            // If start from workstation, next goal is endpoint
            if (std::find(
                    G.workstations.begin(),
                    G.workstations.end(),
                    start_loc) != G.workstations.end())
            {
                this->next_goal_type.push_back("e");
            }
            // Otherwise, next goal is workstation
            else
            {
                this->next_goal_type.push_back("w");
            }
        }

        // Create workstation distribution
        // Number of weight must be the same as number of workstations
        assert(this->G.workstation_weights.size() ==
               this->G.workstations.size());

        // std::random_device rd;
        this->gen = mt19937(this->seed);

        this->workstation_dist = discrete_distribution<int>(
            this->G.workstation_weights.begin(),
            this->G.workstation_weights.end());
        cout << "Workstation distribution: ";
        for (auto w: this->G.workstation_weights)
        {
            cout << w << ", ";
        }
        cout << endl;
    }
}

void KivaSystem::initialize_start_locations()
{
	// Choose random start locations
	// Any non-obstacle locations can be start locations
	// Start locations should be unique
	for (int k = 0; k < num_of_drives; k++)
	{
		int orientation = -1;
		if (consider_rotation)
		{
			orientation = rand() % 4;
		}
		starts[k] = State(G.agent_home_locations[k], 0, orientation);
		paths[k].emplace_back(starts[k]);
		finished_tasks[k].emplace_back(G.agent_home_locations[k], 0);
	}
}

void KivaSystem::initialize_goal_locations()
{
	cout << "Initializing goal locations" << endl;
	if (hold_endpoints || useDummyPaths)
		return;
	// Choose random goal locations
	// Goal locations are not necessarily unique
	for (int k = 0; k < num_of_drives; k++)
	{
		int goal = G.endpoints[rand() % (int)G.endpoints.size()];
		goal_locations[k].emplace_back(goal, 0, 0);
	}
}


int KivaSystem::sample_workstation()
{
    // Sample a workstation based the given weight
    int idx = this->workstation_dist(this->gen);

    return this->G.workstations[idx];
}


int KivaSystem::gen_next_goal(int agent_id, bool repeat_last_goal)
{

	int next = -1;
	if (G.get_r_mode())
	{
		next = G.endpoints[rand() % (int)G.endpoints.size()];
	}
	// Under w mode, alternate goal locations between workstations and endpoints
	else if(G.get_w_mode())
	{
		if (this->next_goal_type[agent_id] == "w")
		{
			if (repeat_last_goal)
			{
				next = G.endpoints[rand() % (int)G.endpoints.size()];
			}
			else
			{
				next = sample_workstation();
				this->next_goal_type[agent_id] = "e";
			}
		}
		else if (this->next_goal_type[agent_id] == "e")
		{
			if (repeat_last_goal)
			{
				next = sample_workstation();
			}
			else
			{
				next = G.endpoints[rand() % (int)G.endpoints.size()];
				this->next_goal_type[agent_id] = "w";
			}
		}
	}

	return next;
}

void KivaSystem::update_goal_locations()
{
    if (!this->LRA_called)
        new_agents.clear();
	if (hold_endpoints)
	{
		unordered_map<int, int> held_locations; // <location, agent id>
		for (int k = 0; k < num_of_drives; k++)
		{
			int curr = paths[k][timestep].location; // current location
			if (goal_locations[k].empty())
			{
				int next = this->gen_next_goal(k);
				while (next == curr || held_endpoints.find(next) != held_endpoints.end())
				{
					next = this->gen_next_goal(k, true);
				}
				goal_locations[k].emplace_back(next, 0, 0);
				held_endpoints.insert(next);
			}
			if (paths[k].back().location == std::get<0>(goal_locations[k].back()) && // agent already has paths to its goal location
				paths[k].back().timestep >= std::get<1>(goal_locations[k].back()))  // after its release time
			{
				int agent = k;
				int loc = std::get<0>(goal_locations[k].back());
				auto it = held_locations.find(loc);
				while (it != held_locations.end()) // its start location has been held by another agent
				{
					int removed_agent = it->second;
					if (std::get<0>(goal_locations[removed_agent].back()) != loc)
						cout << "BUG" << endl;
					new_agents.remove(removed_agent); // another agent cannot move to its new goal location
					cout << "Agent " << removed_agent << " has to wait for agent " << agent << " because of location " << loc << endl;
					held_locations[loc] = agent; // this agent has to keep holding this location
					agent = removed_agent;
					loc = paths[agent][timestep].location; // another agent's start location
					it = held_locations.find(loc);
				}
				held_locations[loc] = agent;
			}
			else // agent does not have paths to its goal location yet
			{
				if (held_locations.find(std::get<0>(goal_locations[k].back())) == held_locations.end()) // if the goal location has not been held by other agents
				{
					held_locations[std::get<0>(goal_locations[k].back())] = k; // hold this goal location
					new_agents.emplace_back(k);							// replan paths for this agent later
					continue;
				}
				// the goal location has already been held by other agents
				// so this agent has to keep holding its start location instead
				int agent = k;
				int loc = curr;
				cout << "Agent " << agent << " has to wait for agent " << held_locations[std::get<0>(goal_locations[k].back())] << " because of location " << std::get<0>(goal_locations[k].back()) << endl;
				auto it = held_locations.find(loc);
				while (it != held_locations.end()) // its start location has been held by another agent
				{
					int removed_agent = it->second;
					if (std::get<0>(goal_locations[removed_agent].back()) != loc)
						cout << "BUG" << endl;
					new_agents.remove(removed_agent); // another agent cannot move to its new goal location
					cout << "Agent " << removed_agent << " has to wait for agent " << agent << " because of location " << loc << endl;
					held_locations[loc] = agent; // this agent has to keep holding its start location
					agent = removed_agent;
					loc = paths[agent][timestep].location; // another agent's start location
					it = held_locations.find(loc);
				}
				held_locations[loc] = agent; // this agent has to keep holding its start location
			}
		}
	}
	else
	{
        if (useDummyPaths)
        {
			if (G.get_w_mode())
			{
				// Assign a goal and assume the agents will hold it.
				for (int k = 0; k < num_of_drives; k++)
				{
					int curr = paths[k][timestep].location; // current location
					// Update goal if there is no goal left or the only goal left
					// is the dummy goal.
					if(goal_locations[k].empty() ||
					(goal_locations[k].size() == 1 &&
					std::get<1>(goal_locations[k][0]) == -2))
					{
						// Empty the current dummy goal. If any.
						if (!goal_locations[k].empty())
						{
							goal_locations[k].clear();
						}

						int next_goal = this->gen_next_goal(k);
						while (next_goal == curr)
						{
							next_goal = this->gen_next_goal(k, true);
						}
						if(screen == 2)
						{
							cout << "Next goal for agent " << k << ": "
								<< next_goal << endl;
						}
						// -1 for actual goal.
						goal_locations[k].emplace_back(next_goal, -1, 0);
						new_agents.emplace_back(k);

						// // If the current goal is workstation, we do not want the
						// // current agent to hold it.
						// if (this->next_goal_type[k] == "e")
						// {
						//     goal_locations[k].emplace_back(
						//         G.endpoints[rand() % (int)G.endpoints.size()], 0);
						// }
					}
				}

				// Change the hold location if it conflicts with any goals of any
				// other agents.
				for (int k = 0; k < num_of_drives; k++)
				{
					int curr_held_loc = std::get<0>(goal_locations[k].back());

					// Obtain all goals locs of all other agents.
					std::vector<int> to_avoid;
					for (int j = 0; j < num_of_drives; j++)
					{
						if (k != j)
						{
							auto agent_goals = goal_locations[j];
							for (tuple<int, int, int> goal : agent_goals)
							{
								to_avoid.push_back(std::get<0>(goal));
							}
						}
					}

					// Check if curr_held_loc is in to_avoid.
					// Found
					if (std::find(to_avoid.begin(), to_avoid.end(), curr_held_loc)
							!= to_avoid.end())
					{
						// Remove the held loc if current agent has two goals
						if (goal_locations[k].size() == 2)
						{
							goal_locations[k].pop_back();
						}

						// Sample a new endpoint that does not conflict with any
						// locs in to_avoid as the new held location.
						int new_held = G.endpoints[
								rand() % (int)G.endpoints.size()];
						while (std::find(to_avoid.begin(), to_avoid.end(), new_held)
							!= to_avoid.end())
						{
							new_held = G.endpoints[
								rand() % (int)G.endpoints.size()];
						}

						if(screen == 2)
						{
							cout << "Next held loc for agent " << k
								<< " conflicts with current goals. "
								<< "Sampled a new held loc "
								<< new_held << endl;
						}

						// -2 for dummy goal.
						goal_locations[k].emplace_back(new_held, -2, 0);

						// Add current agent to replanning set if not already.
						if (std::find(new_agents.begin(), new_agents.end(), k)
							== new_agents.end())
						{
							new_agents.emplace_back(k);
						}

					}
				}

				// Planner assume new_agents to be non-decreasing order.
				new_agents.sort();
			}

			else if (G.get_r_mode())
			{
				for (int k = 0; k < num_of_drives; k++)
				{
					int curr = paths[k][timestep].location; // current location
					if (goal_locations[k].empty())
					{
						goal_locations[k].emplace_back(
                            G.agent_home_locations[k], 0, 0);
					}
					if (goal_locations[k].size() == 1)
					{
						int next;
						do {
							next = G.endpoints[rand() % (int)G.endpoints.size()];
						} while (next == curr);
						goal_locations[k].emplace(
                            goal_locations[k].begin(), next, 0, 0);
						new_agents.emplace_back(k);
					}
				}
			}
        }

        // RHCR Algorithm
        else
        {
            for (int k = 0; k < num_of_drives; k++)
            {
                int curr = paths[k][timestep].location; // current location
				tuple<int, int, int> goal; // The last goal location
				if (goal_locations[k].empty())
				{
					goal = make_tuple(curr, 0, 0);
				}
				else
				{
					goal = goal_locations[k].back();
				}
				double min_timesteps = G.get_Manhattan_distance(
                    std::get<0>(goal), curr);
				while (min_timesteps <= simulation_window)
				// The agent might finish its tasks during the next planning horizon
				{
					// assign a new task
					tuple<int, int, int> next;
					if (G.types[std::get<0>(goal)] == "Endpoint" ||
						G.types[std::get<0>(goal)] == "Workstation")
					{
                        next = make_tuple(this->gen_next_goal(k), 0, 0);
                        while (next == goal)
						{
							next = make_tuple(
                                this->gen_next_goal(k, true), 0, 0);
						}
					}
					else
					{
						std::cout << "ERROR in update_goal_function()" << std::endl;
						std::cout << "The fiducial type should not be " << G.types[curr] << std::endl;
						exit(-1);
					}
					goal_locations[k].emplace_back(next);
					min_timesteps += G.get_Manhattan_distance(std::get<0>(next), std::get<0>(goal));
					goal = next;
				}
			}
		}
	}
}

tuple<vector<double>, double, double> KivaSystem::edge_pair_usage_mean_std(
    vector<vector<double>> &edge_usage)
{
    // For each pair of valid edges (i, j) and (j, i), calculate the absolute
    // of their edge usage difference, and calculate mean and std.
    vector<double> edge_pair_usage(this->G.get_n_valid_edges() / 2, 0.0);
    int valid_edge_id = 0;
    int n_vertices = this->G.rows * this->G.cols;
    for (int i = 0; i < n_vertices; i++)
	{
        // Start from i+1 to ignore wait actions
        for (int j = i + 1; j < n_vertices; j++)
        {
            if (this->G.types[i] != "Obstacle" &&
                this->G.types[j] != "Obstacle" &&
                this->G.get_Manhattan_distance(i, j) <= 1)
            {
                edge_pair_usage[valid_edge_id] = std::abs(
                    edge_usage[i][j] - edge_usage[j][i]);
                valid_edge_id += 1;
            }
        }
	}
    // cout << "Number of valid edge pairs: " << edge_pair_usage.size() << endl;
    // cout << "End of valid edge pair counter: " << valid_edge_id << endl;
    double mean, std;
    std::tie(mean, std) = helper::mean_std(edge_pair_usage);
    return make_tuple(edge_pair_usage, mean, std);
}

tuple<vector<vector<vector<double>>>, vector<vector<double>>> KivaSystem::convert_edge_usage(vector<vector<double>> &edge_usage)
{
    // [h, w, 4] matrix that stores the usage of each edge in the direction of
    // right, up, left, down
    int h = this->G.rows;
    int w = this->G.cols;
    vector<vector<vector<double>>> edge_usage_matrix(
        h, vector<vector<double>>(w, vector<double>(4, 0.0)));
    // [h, w] matrix that stores the usage of wait action in each vertex
    vector<vector<double>> vertex_wait_matrix(h, vector<double>(w, 0.0));

    // Transform edge usage to matrices above
    int n_vertices = this->G.rows * this->G.cols;
    for (int i = 0; i < n_vertices; i++)
	{
        // Start from i+1 to ignore wait actions
        for (int j = 0; j < n_vertices; j++)
        {
            int row_ = this->G.getRowCoordinate(i);
            int col_ = this->G.getColCoordinate(i);

            // Increment wait action usage matrix
            if (i == j)
            {
                vertex_wait_matrix[row_][col_] = edge_usage[i][j];
            }
            // Otherwise, if the edge is valid, update edge usage matrix
            else if (this->G.types[i] != "Obstacle" &&
                this->G.types[j] != "Obstacle" &&
                this->G.get_Manhattan_distance(i, j) <= 1)
            {
                // Obtain direction of edge (i, j)
                int dir = this->G.get_direction(i, j);

                // Set corresponding entry of edge usage matrix
                edge_usage_matrix[row_][col_][dir] = edge_usage[i][j];
            }
        }
	}
    return make_tuple(edge_usage_matrix, vertex_wait_matrix);
}

json KivaSystem::simulate(int simulation_time)
{
	std::cout << "*** Simulating " << seed << " ***" << std::endl;
	this->simulation_time = simulation_time;
	initialize();

    std::vector<std::vector<int>> tasks_finished_timestep;

    bool congested_sim = false;

	for (; timestep < simulation_time; timestep += simulation_window)
	{
		if (this->screen > 0)
			std::cout << "Timestep " << timestep << std::endl;

		update_start_locations();
		update_goal_locations();
		solve();

		// move drives
		auto new_finished_tasks = move();
		if (this->screen > 0)
			std::cout << new_finished_tasks.size() << " tasks has been finished" << std::endl;

		// update tasks
        int n_tasks_finished_per_step = 0;
		for (auto task : new_finished_tasks)
		{
			int id, loc, t;
			std::tie(id, loc, t) = task;
			this->finished_tasks[id].emplace_back(loc, t);
			this->num_of_tasks++;
            n_tasks_finished_per_step++;
			if (this->hold_endpoints)
				this->held_endpoints.erase(loc);
		}

        std::vector<int> curr_task_finished {
            n_tasks_finished_per_step, timestep};
        tasks_finished_timestep.emplace_back(curr_task_finished);

		if (congested())
		{
			cout << "***** Timestep " << timestep
                 << ": Too many traffic jams *****" << endl;
            congested_sim = true;
            if (this->stop_at_traffic_jam)
            {
                break;
            }
		}

        // Overtime?
        double runtime = (double)(clock() - this->start_time)/ CLOCKS_PER_SEC;
        if (runtime >= this->overall_time_limit)
        {
            cout << "***** Timestep " << timestep << ": Overtime *****" << endl;
            break;
        }
	}

	// Compute objective
	double throughput = (double)this->num_of_tasks / this->simulation_time;

	// Compute measures:
	// 1. Variance of tile usage
	// 2. Average number of waiting agents at each timestep
	// 3. Average distance of the finished tasks
    // 4. Average number of turns over the agents
    // 5. Average number of "reversed" action (not going in the direction
    //    suggested by highway) in each timestep, weighted by absolute value of
    //    the difference of edges both directions
    // 6. Variance of edge usage weighted by absolute value of the difference
    //    of edges both directions
    int n_vertices = this->G.rows * this->G.cols;
	std::vector<double> tile_usage(n_vertices, 0.0);
    // Adj matrix: [i,j] stores edge usage of edge from node_i to node_j
    std::vector<vector<double>> edge_usage(
        n_vertices, std::vector<double>(n_vertices, 0.0));
	std::vector<double> num_wait(this->simulation_time, 0.0);
	std::vector<double> finished_task_len;
    std::vector<double> num_turns(num_of_drives, 0.0);
    std::vector<double> num_rev_action(this->simulation_time, 0.0);
	for (int k = 0; k < num_of_drives; k++)
	{
		int path_length = this->paths[k].size();
		for (int j = 0; j < path_length; j++)
		{
			State s = this->paths[k][j];

			// Count tile usage
			tile_usage[s.location] += 1.0;

			if (j < path_length - 1)
			{
				State next_s = this->paths[k][j + 1];
                // See if action is stay
				if (s.location == next_s.location)
				{
                    // The planned path might go beyond the simulation window.
					if (s.timestep < this->simulation_time)
					{
						num_wait[s.timestep] += 1.0;
					}
				}

                // Edge weight from s_t to s_t+1
                double t_to_t_p_1 = this->G.get_weight(
                    s.location, next_s.location);
                // Edge weight from s_t+1 to s_t
                double t_p_1_to_t = this->G.get_weight(
                    next_s.location, s.location);

                // Increment edge usage.
                // This DOES include number of wait action at each vertex (the
                // diagnal of the matrix), but they will be ignored while
                // calculating the mean and std.
                edge_usage[s.location][next_s.location] += 1.0;

                // See if the action follows the edge direction "suggested" by
                // the highway system.
                // If the edge weight that the agent is traversing through is
                // larger than the direction in reverse, then the agent is NOT
                // following the suggested direction.
                if (s.timestep < this->simulation_time &&
                    s.location != next_s.location && t_to_t_p_1 > t_p_1_to_t)
                {
                    num_rev_action[s.timestep] += std::abs(t_to_t_p_1 - t_p_1_to_t);
                }
			}

            // Count the number of turns
            // See if the previous state and the next state change different
            // axis
            if (j > 0 && j < path_length - 1)
            {
                State prev_state = this->paths[k][j-1];
                State next_state = this->paths[k][j+1];
                int prev_row = this->G.getRowCoordinate(prev_state.location);
                int next_row = this->G.getRowCoordinate(next_state.location);
                int prev_col = this->G.getColCoordinate(prev_state.location);
                int next_col = this->G.getColCoordinate(next_state.location);
                if (prev_row != next_row && prev_col != next_col)
                {
                    num_turns[k] += 1;
                }
            }
		}

		int prev_t = 0;
		for (auto task : this->finished_tasks[k])
		{
			if (task.second != 0)
			{
				int curr_t = task.second;
				Path p = this->paths[k];

				// Calculate length of the path associated with this task
				double task_path_len = 0.0;
				for (int t = prev_t; t < curr_t - 1; t++)
				{
					if (p[t].location != p[t + 1].location)
					{
						task_path_len += 1.0;
					}
				}
				finished_task_len.push_back(task_path_len);
				prev_t = curr_t;
			}
		}
	}

    // Longest common sub-path
    vector<int> subpath = helper::longest_common_subpath(
        this->paths, this->simulation_time);

	// Post process data
	// Normalize tile usage s.t. they sum to 1
	double tile_usage_sum = helper::sum(tile_usage);
	helper::divide(tile_usage, tile_usage_sum);

	double tile_usage_mean, tile_usage_std;
	double edge_pair_usage_mean, edge_pair_usage_std;
    vector<double> edge_pair_usage;
    double num_wait_mean, num_wait_std;
	double finished_len_mean, finished_len_std;
    double num_turns_mean, num_turns_std;
    double num_rev_action_mean, num_rev_action_std;
    double avg_task_len = this->G.get_avg_task_len(this->G.heuristics);
    vector<vector<vector<double>>> edge_usage_matrix;
    vector<vector<double>> vertex_wait_matrix;

	std::tie(tile_usage_mean, tile_usage_std) = helper::mean_std(tile_usage);
    std::tie(edge_pair_usage, edge_pair_usage_mean, edge_pair_usage_std) = edge_pair_usage_mean_std(edge_usage);
	std::tie(num_wait_mean, num_wait_std) = helper::mean_std(num_wait);
	std::tie(finished_len_mean, finished_len_std) = helper::mean_std(finished_task_len);
    std::tie(num_turns_mean, num_turns_std) = helper::mean_std(num_turns);
    std::tie(num_rev_action_mean, num_rev_action_std) = helper::mean_std(num_rev_action);
    std::tie(edge_usage_matrix, vertex_wait_matrix) = convert_edge_usage(edge_usage);

	// Log some of the results
	std::cout << std::endl;
	std::cout << "Throughput: " << throughput << std::endl;
	std::cout << "Std of tile usage: " << tile_usage_std << std::endl;
    std::cout << "Std of edge pair usage: " << edge_pair_usage_std << std::endl;
	std::cout << "Average wait at each timestep: " << num_wait_mean << std::endl;
	std::cout << "Average path length of each finished task: " << finished_len_mean << std::endl;
    std::cout << "Average path length of each task: " << avg_task_len << std::endl;
    std::cout << "Average number of turns: " << num_turns_mean << std::endl;
    std::cout << "Average number of reversed actions in highway: " << num_rev_action_mean << std::endl;
    std::cout << "Length of longest common path: " << subpath.size()
              << std::endl;

	update_start_locations();
	std::cout << std::endl
			  << "Done!" << std::endl;
	save_results();

	// Create the result json object
	json result;
	result = {
		{"throughput", throughput},
		{"tile_usage", tile_usage},
        {"edge_pair_usage", edge_pair_usage},
        {"edge_usage_matrix", edge_usage_matrix},
        {"vertex_wait_matrix", vertex_wait_matrix},
		{"num_wait", num_wait},
        {"num_turns", num_turns},
        {"num_rev_action", num_rev_action},
		{"finished_task_len", finished_task_len},
		{"tile_usage_mean", tile_usage_mean},
		{"tile_usage_std", tile_usage_std},
        {"edge_pair_usage_mean", edge_pair_usage_mean},
        {"edge_pair_usage_std", edge_pair_usage_std},
		{"num_wait_mean", num_wait_mean},
		{"num_wait_std", num_wait_std},
		{"finished_len_mean", finished_len_mean},
		{"finished_len_std", finished_len_std},
        {"num_turns_mean", num_turns_mean},
        {"num_turns_std", num_turns_std},
        {"num_rev_action_mean", num_rev_action_mean},
        {"num_rev_action_std", num_rev_action_std},
        {"tasks_finished_timestep", tasks_finished_timestep},
        {"avg_task_len", avg_task_len},
        {"congested", congested_sim},
        {"longest_common_path", subpath},
	};
	return result;
}
