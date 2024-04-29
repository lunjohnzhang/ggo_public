import os
import json
import fire
import numpy as np

from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from env_search.utils import (
    kiva_obj_types,
    competition_obj_types,
    kiva_env_str2number,
    read_in_kiva_map,
    competition_env_str2number,
    read_in_competition_map,
    get_Manhattan_distance,
    loc_to_idx,
    idx_to_loc,
    get_n_valid_edges,
    k_largest_index_argpartition,
    get_direction,
    kiva_compress_edge_weights,
    write_map_str_to_json,
    adj_matrix_to_edge_weights_matrix,
    min_max_normalize,
    MAP_DIR,
)
from env_search.competition.update_model.utils import (
    comp_compress_edge_matrix, Map)


def run_shortest_path(graph, start, goal):
    """Run Dijkstra's algo and return the shortest path from start to goal.

    Args:
        graph (np.ndarray): size [n_node, n_node], describe the graph
        start (int): start index
        goal (int): goal index
    """
    # fire.Fire(gen_expert_baseline)
    graph = csr_matrix(graph)
    dist_matrix, predecessors, sources = dijkstra(csgraph=graph,
                                                  directed=True,
                                                  indices=start,
                                                  return_predecessors=True,
                                                  min_only=True)
    result_path = reconstruct_path(predecessors, goal)
    return result_path


def reconstruct_path(predecessors, goal):
    # No solution
    if predecessors[goal] == -9999:
        return None

    path = [goal]
    curr_node = goal
    while predecessors[curr_node] != -9999:
        path.append(predecessors[curr_node])
        curr_node = predecessors[curr_node]
    path.reverse()
    return path


def sample_new_tasks(all_start_loc, all_goal_loc, num_tasks):
    # One random (non-overlap) start and goal for each task
    starts = np.random.choice(all_start_loc, num_tasks, replace=False)

    # For each start, sample a goal that is not the same as start
    goals = []
    for start in starts:
        goal = np.random.choice(all_goal_loc, 1)[0]
        while start == goal:
            goal = np.random.choice(all_goal_loc, 1)[0]
        goals.append(goal)
    goals = np.array(goals)
    assert np.all(starts != goals)
    return starts, goals


def write_edge_weights(
    map_np,
    map_str,
    edge_weight_matrix,
    domain,
    map_filepath,
    store_dir,
    save_map_name,
):
    if domain == "kiva":
        edge_weights = kiva_compress_edge_weights(
            map_np, edge_weight_matrix, block_idxs=[kiva_obj_types.index("@")])
    elif domain == "competition":
        comp_map = Map(map_filepath)
        edge_weights = comp_compress_edge_matrix(comp_map,
                                                 edge_weight_matrix.flatten())

    if "flow_baseline" in save_map_name:
        edge_weights = min_max_normalize(edge_weights, 0.1, 100).tolist()

    write_map_str_to_json(
        os.path.join(store_dir, f"{save_map_name}.json"),
        map_str,
        save_map_name,
        domain,
        weight=True,
        weights=[1.0, *edge_weights],
        optimize_wait=False,
    )


def gen_expert_baseline(
    map_filepath,
    # num_agents,
    domain,
    n_iter=100_000,
    store_dir=MAP_DIR,
    w_mode=True,
):
    """Generate human baseline map based on single agent paths on unweighted
    map, ignoring collisions.

    Reference:
        Chen et.al, "Traffic Flow Optimisation for Lifelong Multi-Agent Path
            Finding", 2023
        Liron et.al, "Improved Solvers for Bounded-Suboptimal Multi-Agent Path
            Finding", 2016
    """
    if domain == "kiva":
        map_str, map_name = read_in_kiva_map(map_filepath)
        map_np = kiva_env_str2number(map_str)
        block_idx = [kiva_obj_types.index("@")]
        start_idx = [
            # kiva_obj_types.index("."),
            kiva_obj_types.index("e"),
            # kiva_obj_types.index("w"),
        ]
        if w_mode:
            goal_idx = [kiva_obj_types.index("w")]
        else:
            goal_idx = [kiva_obj_types.index("e")]
    elif domain == "competition":
        map_str, map_name = read_in_competition_map(map_filepath)
        map_np = competition_env_str2number(map_str)
        block_idx = [competition_obj_types.index("@")]
        start_idx = [competition_obj_types.index(".")]
        goal_idx = [competition_obj_types.index(".")]
    elif domain == "competition_kiva":
        map_str, map_name = read_in_competition_map(map_filepath)
        map_np = competition_env_str2number(map_str)
        block_idx = [competition_obj_types.index("@")]
        start_idx = [
            competition_obj_types.index("e"),
            # competition_obj_types.index("w"),
        ]
        if w_mode:
            goal_idx = [competition_obj_types.index("w")]
        else:
            goal_idx = [competition_obj_types.index("e")]
        domain = "competition"  # For saving

    h, w = map_np.shape
    n_valid_edges = get_n_valid_edges(map_np, bi_directed=True, domain=domain)

    # All potential start and goal locations
    all_start_idx = []
    for idx in start_idx:
        starts = np.argwhere(map_np == idx)
        all_start_idx.append(starts)
    all_start_idx = np.concatenate(all_start_idx, axis=0)

    all_goal_idx = []
    for idx in goal_idx:
        goals = np.argwhere(map_np == idx)
        all_goal_idx.append(goals)
    all_goal_idx = np.concatenate(all_goal_idx, axis=0)

    # assert num_agents <= all_task_idx.shape[0]

    # Convert 2D loc index to node index (0 to h * w - 1)
    all_start_loc = []
    for idx in all_start_idx:
        all_start_loc.append(idx_to_loc(idx, w))

    all_goal_loc = []
    for idx in all_goal_idx:
        all_goal_loc.append(idx_to_loc(idx, w))

    # Initialize graph. Start with all valid edges being 1, set invalid
    # edges to 0
    n_nodes = h * w
    graph_hm = np.zeros((n_nodes, n_nodes))  # aka edge weight
    graph_tf = np.zeros((n_nodes, n_nodes))  # aka edge weight
    for node_i in range(h * w):
        i_x, i_y = loc_to_idx(node_i, w)

        # Skip if node i is block
        if map_np[i_x, i_y] in block_idx:
            continue

        for node_j in range(h * w):
            j_x, j_y = loc_to_idx(node_j, w)

            # Skip if node j is block
            if map_np[j_x, j_y] in block_idx:
                continue

            # check if edge[i,j] is valid
            if get_Manhattan_distance(node_i, node_j, w) <= 1:
                graph_hm[node_i, node_j] = 1
                graph_tf[node_i, node_j] = 1

    # Run iterative routine
    tile_usage_matrix = np.zeros((h, w))

    # Remember invalid edges, same for hm and tf
    invalid_edge_idx = np.argwhere(graph_hm == 0)

    ########### Liron 2016 ###########
    alpha = 0.5
    beta = 1.2
    gamma = 1.3
    edge_use = np.zeros((n_nodes, n_nodes))  # n
    flow_pref = np.zeros((n_nodes, n_nodes))  # p
    interference_cost = np.zeros((n_nodes, n_nodes))  # t
    saturation_cost = np.zeros((n_nodes, n_nodes))  # s
    hm_cost = np.zeros((n_nodes, n_nodes))  # c

    for k in tqdm(range(n_iter)):
        starts, goals = sample_new_tasks(all_start_loc,
                                         all_goal_loc,
                                         num_tasks=1)

        for start, goal in zip(starts, goals):
            path = run_shortest_path(graph_hm, start, goal)
            assert path is not None

            # Update stats
            for i, node in enumerate(path):
                x, y = loc_to_idx(node, w)
                tile_usage_matrix[x, y] += 1

                if i < len(path) - 1:
                    next_node = path[i + 1]
                    edge_use[node, next_node] += 1

        edge_use_rev = edge_use.T
        flow_pref = alpha * edge_use / n_iter
        interference_cost = beta * edge_use_rev / n_iter
        saturation_cost = gamma**((edge_use + edge_use_rev) / (2 * n_iter))
        hm_cost = 1 - flow_pref + interference_cost + saturation_cost
        graph_hm = hm_cost

        # Set invalid edges as 0
        graph_hm[invalid_edge_idx[:, 0], invalid_edge_idx[:, 1]] = 0
        # Also ignore edges going towards each node itself
        graph_hm[np.arange(n_nodes), np.arange(n_nodes)] = 0

    # Ignoring invalid edges, take the smallest 1/7 hm cost edges as the
    # highways (lower cost), see paper for detail.
    n_highway_edges = int(1 / 7 * n_valid_edges)
    hm_cost[invalid_edge_idx[:, 0], invalid_edge_idx[:, 1]] = np.inf
    # Also ignore edges going towards each node itself
    hm_cost[np.arange(n_nodes), np.arange(n_nodes)] = np.inf
    top_k_hm_edges = k_largest_index_argpartition(-hm_cost, n_highway_edges)
    # Choose 1/5th randomly
    hwy_edge_idx = np.random.choice(np.arange(len(top_k_hm_edges)),
                                    int(1 / 5 * len(top_k_hm_edges)),
                                    replace=False)
    top_k_hm_edges = top_k_hm_edges[hwy_edge_idx]

    # Create highway
    edge_weight_matrix = np.ones((h, w, 4))  # right, up, left, down
    for edge_idx in top_k_hm_edges:
        i, j = edge_idx
        x, y = loc_to_idx(i, w)
        dir = get_direction(i, j, w)
        edge_weight_matrix[x, y, dir] = 0.5

    save_map_name = f"{map_name}_HM_baseline"
    write_edge_weights(
        map_np,
        map_str,
        edge_weight_matrix,
        domain,
        map_filepath,
        store_dir,
        save_map_name,
    )

    ########### Chen 2023 ###########
    tile_usage_matrix = np.zeros((h, w))
    edge_use = np.zeros((n_nodes, n_nodes))
    for k in tqdm(range(n_iter)):
        starts, goals = sample_new_tasks(all_start_loc,
                                         all_goal_loc,
                                         num_tasks=1)

        for start, goal in zip(starts, goals):
            path = run_shortest_path(graph_tf, start, goal)
            assert path is not None

            # Update stats
            for i, node in enumerate(path):
                x, y = loc_to_idx(node, w)
                tile_usage_matrix[x, y] += 1

                if i < len(path) - 1:
                    next_node = path[i + 1]
                    edge_use[node, next_node] += 1

        # p_v
        p_v = np.ceil((tile_usage_matrix - 1) / 2)
        # Contraflow
        edge_use_rev = edge_use.T
        C_e = np.multiply(edge_use, edge_use_rev)
        # Edge weights
        # edge_weights_graph = np.zeros((n_nodes, n_nodes))
        # for i in range(n_nodes):
        #     for j in range(n_nodes):
        #         v_2_x, v_2_y = loc_to_idx(j, w)
        #         edge_weights_graph[i, j] = 1 + C_e[i, j] + p_v[v_2_x, v_2_y]
        # graph_tf = edge_weights_graph
        # assert np.all(1 + C_e + p_v.flatten() == graph_tf)
        graph_tf = 1 + C_e + p_v.flatten()

        # Set invalid edges as 0
        graph_tf[invalid_edge_idx[:, 0], invalid_edge_idx[:, 1]] = 0
        # Also ignore edges going towards each node itself
        graph_tf[np.arange(n_nodes), np.arange(n_nodes)] = 0

    # Transform graph to edge weights matrix
    edge_weight_matrix = adj_matrix_to_edge_weights_matrix(graph_tf, h, w)

    save_map_name = f"{map_name}_flow_baseline"
    write_edge_weights(
        map_np,
        map_str,
        edge_weight_matrix,
        domain,
        map_filepath,
        store_dir,
        save_map_name,
    )


if __name__ == "__main__":
    fire.Fire(gen_expert_baseline)
