"""Miscellaneous project-wide utilities."""
import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from env_search import MAP_DIR

# 6 object types for kiva map:
# '.' (0) : empty space
# '@' (1): obstacle (shelf)
# 'e' (2): endpoint (point around shelf)
# 'r' (3): robot start location (not searched)
# 's' (4): one of 'r'
# 'w' (5): workstation
# NOTE:
# 1: only the first 2 or 3 objects are searched by QD
# 2: s (r_s) is essentially one of r s.t. in milp can make the graph
# connected
kiva_obj_types = ".@ersw"
KIVA_ROBOT_BLOCK_WIDTH = 4
KIVA_WORKSTATION_BLOCK_WIDTH = 2
KIVA_ROBOT_BLOCK_HEIGHT = 4
MIN_SCORE = 0

# right, up, left, down (in that direction because it is the same
# of the simulator!!)
kiva_directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]

kiva_color_map = {
    kiva_obj_types.index("."): "white",
    kiva_obj_types.index("@"): "black",
    kiva_obj_types.index("e"): "deepskyblue",
    kiva_obj_types.index("r"): "orange",
    kiva_obj_types.index("w"): "fuchsia",
}

kiva_dir_map = {
    (0, 1): 0,  # right
    (-1, 0): 1,  # up
    (0, -1): 2,  # left
    (1, 0): 3,  # down
}

kiva_rev_d_map = {
    (0, 1): (0, -1),  # right to left
    (-1, 0): (1, 0),  # up to down
    (0, -1): (0, 1),  # left to right
    (1, 0): (-1, 0),  # down to up
}

# 4 object types for competition map
competition_obj_types = ".@ews"


def format_env_str(env_str):
    """Format the env from List[str] to pure string separated by \n """
    return "\n".join(env_str)


def env_str2number(env_str, obj_types):
    env_np = []
    for row_str in env_str:
        # print(row_str)
        row_np = [obj_types.index(tile) for tile in row_str]
        env_np.append(row_np)
    return np.array(env_np, dtype=int)


def get_project_dir():
    env_search_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(env_search_dir)


def env_number2str(env_np, obj_types):
    env_str = []
    n_row, n_col = env_np.shape
    for i in range(n_row):
        curr_row = []
        for j in range(n_col):
            curr_row.append(obj_types[env_np[i, j]])
        env_str.append("".join(curr_row))
    return env_str


def kiva_env_str2number(env_str):
    """
    Convert kiva env in string format to np int array format.

    Args:
        env_str (List[str]): kiva env in string format

    Returns:
        env_np (np.ndarray)
    """
    return env_str2number(env_str, kiva_obj_types)


def kiva_env_number2str(env_np):
    """
    Convert kiva env in np int array format to str format.

    Args:
        env_np (np.ndarray): kiva env in np array format

    Returns:
        env_str (List[str])
    """
    return env_number2str(env_np, kiva_obj_types)


def clean_competition_str_env(raw_env_str):
    """Replace arbitrary obstacle chars in raw maps with `@`.

    Args:
        raw_env_str (List[str]): raw competition env in string format

    Returns:
        env_str (List[str])
    """
    env_str = []
    for row in raw_env_str:
        clean_row = ""
        for c in row:
            # for warehouse map in competition
            if c == 'S' or c == 'E':
                clean_row += '.'
            # e and w don't change
            elif c == 'e' or c == 'w':
                clean_row += c
            elif c != '.':
                clean_row += '@'
            else:
                clean_row += '.'
        env_str.append(clean_row)
    return env_str


def competition_env_str2number(env_str):
    """
    Convert competition env in string format to np int array format.

    Args:
        env_str (List[str]): competition env in string format

    Returns:
        env_np (np.ndarray)
    """
    # Replace all obstacle chars with '@'
    env_str = clean_competition_str_env(env_str)
    return env_str2number(env_str, competition_obj_types)


def competition_env_number2str(env_np):
    """
    Convert competition env in np int array format to str format.

    Args:
        env_np (np.ndarray): competition env in np array format

    Returns:
        env_str (List[str])
    """
    return env_number2str(env_np, competition_obj_types)


def flip_one_r_to_s(env_np, obj_types=kiva_obj_types):
    """
    Change one of 'r' in the env to 's' for milp
    """
    all_r = np.argwhere(env_np == obj_types.index("r"))
    if len(all_r) == 0:
        raise ValueError("No 'r' found")
    to_replace = all_r[0]
    env_np[tuple(to_replace)] = obj_types.index('s')
    return env_np


def flip_one_e_to_s(env_np, obj_types=kiva_obj_types):
    """
    Change one of 'e' in the env to 's' for milp
    """
    all_e = np.argwhere(env_np == obj_types.index("e"))
    if len(all_e) == 0:
        raise ValueError("No 'e' found")
    to_replace = all_e[0]
    env_np[tuple(to_replace)] = obj_types.index('s')
    return env_np


def flip_one_empty_to_s(env_np, obj_types=kiva_obj_types):
    all_empty = np.argwhere(env_np == obj_types.index("."))
    if len(all_empty) == 0:
        raise ValueError("No '.' found")
    to_replace = all_empty[0]
    env_np[tuple(to_replace)] = obj_types.index('s')
    return env_np


def flip_tiles(env_np, from_tile, to_tile, obj_types=kiva_obj_types):
    """Replace ALL occurance of `from_tile` to `flip_target` in `to_tile"""
    all_from_tiles = np.where(env_np == obj_types.index(from_tile))
    if len(all_from_tiles[0]) == 0:
        raise ValueError(f"No '{from_tile}' found")
    env_np[all_from_tiles] = obj_types.index(to_tile)
    return env_np


def read_in_kiva_map(map_filepath):
    """
    Read in kiva map and return in str format
    """
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
        raw_env = raw_env_json["layout"]
        name = raw_env_json["name"]
    return raw_env, name


def read_in_competition_map(map_filepath):
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)
        raw_env = raw_env_json["layout"]
        raw_env = clean_competition_str_env(raw_env)
        name = raw_env_json["name"]
    return raw_env, name


def write_map_str_to_json(
    map_filepath,
    map_str,
    name,
    domain,
    weight=False,
    weights=None,
    milp_runtime=None,
    optimize_wait=False,
    piu_n_agent=None,
):
    """Write specified map to disk.

    Args:
        map_filepath (str): filepath to write
        map_str (List[str]): map in str format
        name (str): name of the map
        domain (str): domain of the map
        weight (bool, optional): Whether the map is weighted. Defaults to False.
        weights (List[float], optional): the edge weights of the map. Defaults
            to None.
        milp_runtime (float, optional): runtime of MILP. Defaults to None.
        optimize_wait (float, optional): the weights optimize wait costs or not.
        piu_n_agent (int, optional): number of agent the weights are generated
            with PIU algorithm.
    """
    to_write = {
        "name": name,
        "layout": map_str,
        "milp_runtime": milp_runtime,
        "optimize_wait": optimize_wait,
        "piu_n_agent": piu_n_agent,
    }
    if domain == "kiva":
        map_np = kiva_env_str2number(map_str)
        to_write["weight"] = weight
        to_write["n_row"] = map_np.shape[0]
        to_write["n_col"] = map_np.shape[1]
        to_write["n_endpoint"] = sum(row.count('e') for row in map_str)
        to_write["n_agent_loc"] = sum(row.count('r') for row in map_str)
        to_write["n_shelf"] = sum(row.count('@') for row in map_str)
        to_write["maxtime"] = 5000
    elif domain == "competition":
        map_np = competition_env_str2number(map_str)
        to_write["weight"] = weight
        to_write["n_row"] = map_np.shape[0]
        to_write["n_col"] = map_np.shape[1]

    to_write["weights"] = weights

    with open(map_filepath, "w") as json_file:
        json.dump(to_write, json_file, indent=4)


def write_iter_update_model_to_json(filepath, model_param, model_type):
    to_write = {
        "type": str(model_type),
        "params": model_param,
    }
    with open(filepath, "w") as f:
        json.dump(to_write, f, indent=4)


def set_spines_visible(ax: plt.Axes):
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_visible(True)


def n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    # params_ = 0
    # state_dict = model.state_dict()
    # for _, param in state_dict.items():
    #     params_ += np.prod(param.shape)
    # print("validate: ", params_)

    return params


def rewrite_map(path, domain):
    if domain == "kiva":
        env, name = read_in_kiva_map(path)
    write_map_str_to_json(path, env, name, domain)


def kiva_get_highway_suggestion_graph(map_np, weights, block_idxs):
    """Get highway suggestion graph `G_h` from the weights of bidirected graph
       `G`.

    The highway suggestion graph G_h has the same nodes of the original graph.
    For each edge e between adjacent nodes u and v in G_h, the dirction of e is
    the same as the one with smaller edge weights between e'(u, v) and
    e''(v, u) in G. The edge weight of e is determined by the absolute value of the difference between e' and e''. Essentially, the direction of e
    encodes the direction `suggested` by the highway and the edge weight
    indicates the `intensity` of the suggestion.

    Args:
        map_np (np.ndarray): the map G
        weights (list): edge weights of G
        block_idxs (list): list of blocking indces

    Returns:
        highway_suggestion_graph (np.ndarray): size (h, w, 4). Each node has a
        entry of size 4 encoding the edge weights in 4 directions (right, up,
        left, down). If the value is np.nan, the edge is invalid.

    """
    h, w = map_np.shape
    optimized_graph = kiva_uncompress_edge_weights(map_np, weights, block_idxs)
    highway_suggestion_graph = copy.deepcopy(optimized_graph)

    for x in range(h):
        for y in range(w):
            # Skip all obstacles
            if map_np[x, y] in block_idxs:
                continue
            for idx, (dx, dy) in enumerate(kiva_directions):
                n_x = x + dx
                n_y = y + dy
                if n_x < h and n_x >= 0 and \
                   n_y < w and n_y >= 0 and \
                   map_np[n_x,n_y] not in block_idxs:
                    # Get the rev_idx of edge in node (n_x, n_y)
                    # rev_xy = rev_d[idx]
                    # rev_idx = kiva_directions.index(rev_xy)
                    rev_idx = get_rev_edge_index((dx, dy), kiva_directions)

                    # print(f"node: ({x},{y}), adj node: ({n_x},{n_y})")
                    # print(f"node to adj: {optimized_graph[x,y,idx]}, adj to node: {optimized_graph[n_x, n_y, rev_idx]}")

                    # Next edge is already determined
                    if np.isnan(highway_suggestion_graph[n_x, n_y, rev_idx]):
                        continue

                    # Compare edge weights
                    suggest_edge_weight = np.abs(
                        optimized_graph[n_x, n_y, rev_idx] - \
                        optimized_graph[x,y,idx])
                    if optimized_graph[x, y, idx] > optimized_graph[n_x, n_y,
                                                                    rev_idx]:
                        highway_suggestion_graph[x, y, idx] = np.nan
                        highway_suggestion_graph[n_x, n_y,
                                                 rev_idx] = suggest_edge_weight
                    else:
                        highway_suggestion_graph[n_x, n_y, rev_idx] = np.nan
                        highway_suggestion_graph[x, y,
                                                 idx] = suggest_edge_weight
    return highway_suggestion_graph


def get_Manhattan_distance(loc1, loc2, w):
    return abs(loc1 // w - loc2 // w) + abs(loc1 % w - loc2 % w)


def loc_to_idx(loc, w):
    """Transform location (a single number) to index (x, y)

    Args:
        loc (int): location
        w (int): width of map
    """
    return loc // w, loc % w


def idx_to_loc(idx, w):
    """Transform index (x, y) to location (a single number)

    Args:
        idx (tuple(int, int)): index
        w (int): width of map
    """
    x, y = idx
    return w * x + y


def get_direction(from_loc, to_loc, w):
    move = [1, -w, -1, w]  # right, up, left, down
    for i in range(4):
        if move[i] == to_loc - from_loc:
            return i
    if from_loc == to_loc:
        return 4
    return -1


def adj_matrix_to_edge_weights_matrix(adj_matrix, h, w):
    """Transform adj matrix to edge matrix of size (h, w, 4)
    Args:
        adj_matrix (np.ndarray): size [n_nodes, n_nodes]
        h (int): height
        w (int): width
    """
    assert adj_matrix.shape[0] == h * w
    edge_weights_matrix = np.zeros((h, w, 4))
    move = [1, -w, -1, w]
    for x in range(h):
        for y in range(w):
            loc = idx_to_loc((x, y), w)
            for idx, dir in enumerate(move):
                n_loc = loc + dir
                if get_Manhattan_distance(loc, n_loc,
                                          w) <= 1 and 0 <= n_loc < h * w:
                    edge_weights_matrix[x, y, idx] = adj_matrix[loc, n_loc]
    return edge_weights_matrix


def kiva_uncompress_edge_weights(
    map_np,
    edge_weights,
    block_idxs,
    fill_value=np.nan,
):
    """Transform the raw list of edge weights to np array of size (h, w, 4),
    where the order in the last dimension is right, up, left, down (in that
    direction because it is the same as the simulator!!)
    Args:
        map_np (np.ndarray): the map in numpy format
        edge_weights (list of float): raw edge weights
        block_idxs: the indices of the map that should be considered as
            obstacles.

    Returns:
        edge_weights_matrix (np.ndarray of size(# nodes, 4)): edge weights of
            the bi-directed graph.
    """
    h, w = map_np.shape
    edge_weights_matrix = np.zeros((h, w, 4))
    edge_weights_matrix[:, :, :] = fill_value
    weight_cnt = 0
    move = [1, -w, -1, w]

    for i in range(h * w):
        x = i // w
        y = i % w
        assert i == w * x + y
        if map_np[x, y] in block_idxs:
            continue
        for dir in range(4):
            n_x = (i + move[dir]) // w
            n_y = (i + move[dir]) % w
            if 0 <= i + move[dir] < h * w and get_Manhattan_distance(
                    i, i + move[dir], w) <= 1 and map_np[n_x,
                                                         n_y] not in block_idxs:
                edge_weights_matrix[x, y, dir] = edge_weights[weight_cnt]
                weight_cnt += 1
    assert weight_cnt == len(edge_weights)
    return edge_weights_matrix

    # right, up, left, down (in that direction because it is the same
    # as the simulator!!)
    # for x in range(h):
    #     for y in range(w):
    #         # Skip all obstacles
    #         if map_np[x,y] in block_idxs:
    #             continue
    #         for idx, (dx, dy) in enumerate(d):
    #             n_x = x + dx
    #             n_y = y + dy
    #             if n_x < h and n_x >= 0 and \
    #                n_y < w and n_y >= 0 and \
    #                map_np[n_x,n_y] not in block_idxs:
    #                 optimized_graph[x,y,idx] = weights[weight_cnt]
    #                 weight_cnt += 1
    # assert weight_cnt == len(weights)
    # return optimized_graph


def kiva_uncompress_wait_costs(
    map_np,
    wait_costs,
    block_idxs,
    fill_value=np.nan,
):
    """Transform the raw list of wait costs to np array of size (h, w)
    Args:
        map_np (np.ndarray): the map in numpy format
        wait_costs (list of float): raw edge weights
        block_idxs: the indices of the map that should be considered as
            obstacles.

    Returns:
        optimized_graph (np.ndarray of size(# nodes, 4)): edge weights of the
            bi-directed graph.
    """
    h, w = map_np.shape
    wait_costs_matrix = np.zeros((h, w))
    wait_costs_matrix[:, :] = fill_value
    i = 0

    for x in range(h):
        for y in range(w):
            if map_np[x, y] in block_idxs:
                continue
            wait_costs_matrix[x, y] = wait_costs[i]
            i += 1
    return wait_costs_matrix


def kiva_compress_edge_weights(map_np, edge_weights, block_idxs):
    """Transform the edge weigths (np array of size (h, w, 4)) to raw weights.
    The order in the last dimension is right, up, left, down (in that
    direction because it is the same as the simulator!!)

    Args:
        edge_weights (np.ndarray of size(h, w, 4)): edge weights of the
            bi-directed graph.

    Returns:
        compress_edge_weights (list of float): raw edge weights
    """
    h, w = map_np.shape
    compress_edge_weights = []
    for x in range(h):
        for y in range(w):
            # Skip all obstacles
            if map_np[x, y] in block_idxs:
                continue
            # Check if neighbor is obstacle or out of range
            # node_idx = w * x + y
            for d_idx, (dx, dy) in enumerate(kiva_directions):
                n_x = x + dx
                n_y = y + dy
                if n_x < h and n_x >= 0 and \
                    n_y < w and n_y >= 0 and \
                    map_np[n_x,n_y] not in block_idxs:
                    # Edge should be valid
                    assert not np.isnan(edge_weights[x][y][d_idx])
                    compress_edge_weights.append(edge_weights[x][y][d_idx])
    return compress_edge_weights


def kiva_compress_wait_costs(map_np, wait_costs, block_idxs):
    """Transform the wait costs (np array of size (h, w, 4)) to raw weights.
    The order in the last dimension is right, up, left, down (in that
    direction because it is the same as the simulator!!)

    Args:
        wait_costs (np.ndarray of size(h, w, 4)): wait costs of the
            bi-directed graph.

    Returns:
        compress_wait_costs (list of float): raw wait costs
    """
    h, w = map_np.shape
    compress_wait_costs = []
    for x in range(h):
        for y in range(w):
            # Skip all obstacles
            if map_np[x, y] in block_idxs:
                continue
            compress_wait_costs.append(wait_costs[x, y])
    return compress_wait_costs


def get_rev_edge_index(direction, d):
    """Get the direction index of the reversion direction in d.

    Args:
        direction: the given direction
        d: all directions, usually d = kiva_directions
    """
    rev_d = kiva_rev_d_map[direction]
    return d.index(rev_d)


def get_n_valid_edges(map_np, bi_directed, domain):
    n_valid_edges = 0
    if domain == "competition":
        block_idxs = [
            competition_obj_types.index("@"),
        ]
    elif domain == "kiva":
        block_idxs = [
            kiva_obj_types.index("@"),
        ]
    d = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    h, w = map_np.shape
    for x in range(h):
        for y in range(w):
            # Skip all obstacles
            if map_np[x, y] in block_idxs:
                continue
            # Check if neighbor is obstacle or out of range
            for dx, dy in d:
                n_x = x + dx
                n_y = y + dy
                if n_x < h and n_x >= 0 and \
                    n_y < w and n_y >= 0 and \
                    map_np[n_x,n_y] not in block_idxs:
                    n_valid_edges += 1
    assert n_valid_edges % 2 == 0
    # If graph is uni-directed, we have half the number of the edges
    # compared to bi-directed counterpart.
    if not bi_directed:
        n_valid_edges = n_valid_edges // 2

    return n_valid_edges


def get_n_valid_vertices(map_np, domain):
    if domain == "competition":
        return np.sum(map_np != competition_obj_types.index("@"), dtype=int)
    elif domain == "kiva":
        return np.sum(map_np != kiva_obj_types.index("@"), dtype=int)


def min_max_normalize(arr, lb, ub):
    """Min-Max normalization on 1D array `arr`

    Args:
        arr (array-like): array to be normalized
        lb (float): lower bound
        ub (float): upper bound
    """
    arr = np.asarray(arr)
    min_ele = np.min(arr)
    max_ele = np.max(arr)
    if max_ele - min_ele < 1e-3:
        # Clip and then return
        return np.clip(arr, lb, ub)
    arr_norm = lb + (arr - min_ele) * (ub - lb) / (max_ele - min_ele)
    if np.any(np.isnan(arr_norm)):
        print(arr)
    return arr_norm


def min_max_normalize_2d(arr, lb, ub):
    """Min-Max normalization on 2D array `arr`

    Args:
        arr (array-like): array to be normalized
        lb (float): lower bound
        ub (float): upper bound
    """
    arr = np.asarray(arr)
    min_sols = np.min(arr, axis=1, keepdims=True)
    max_sols = np.max(arr, axis=1, keepdims=True)
    arr_norm = lb + (arr - min_sols) * (ub - lb) / (max_sols - min_sols)
    return arr_norm


def k_largest_index_argpartition(a, k):
    """largest index of given array `a`

    Reference: https://stackoverflow.com/a/43386556
    """
    idx = np.argpartition(-a.ravel(), k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))


def load_pibt_default_config():
    """Return default PIBT config as json str.
    """
    config_path = "WPPL/configs/pibt_default_no_rot.json"
    with open(config_path) as f:
        config = json.load(f)
        config_str = json.dumps(config)
    return config_str


def single_sim_done(result_dir_full):
    """Check if previous single simulation is done.
    """
    config_file = os.path.join(result_dir_full, "config.json")
    result_file = os.path.join(result_dir_full, "result.json")

    if os.path.exists(config_file) and os.path.exists(result_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            with open(result_file, "r") as f:
                result = json.load(f)
        except json.decoder.JSONDecodeError:
            return False
        return True
    return False
