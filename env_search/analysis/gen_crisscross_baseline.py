import os
import fire
import numpy as np

from env_search.utils import (
    read_in_kiva_map,
    kiva_env_str2number,
    kiva_compress_edge_weights,
    kiva_compress_wait_costs,
    kiva_obj_types,
    kiva_dir_map,
    kiva_directions,
    MAP_DIR,
    write_map_str_to_json,
    kiva_uncompress_edge_weights,
    competition_obj_types,
    competition_env_str2number,
    read_in_competition_map,
    get_n_valid_vertices,
)

from env_search.analysis.visualize_highway import (
    plot_highway_suggest_graph,
    plot_highway_edge_weights,
)

from env_search.competition.update_model.utils import (
    Map,
    comp_compress_edge_matrix,
    comp_compress_vertex_matrix,
)


def valid_edge(map_np, n_x, n_y, block_idxs):
    h, w = map_np.shape
    return n_x < h and n_x >= 0 and \
           n_y < w and n_y >= 0 and \
           map_np[n_x,n_y] not in block_idxs


def gen_crisscross_baseline(
    map_filepath,
    store_dir=MAP_DIR,
    domain="kiva",
    bi_directed=True,
):
    """Generate baseline bi-directed highway map s.t. the directions on each
    row/column will alternate between left and right (for each row) or up and
    down (for each column)
    """
    block_idxs = None
    if domain == "kiva":
        block_idxs = [
            kiva_obj_types.index("@"),
        ]

        base_map_str, map_name = read_in_kiva_map(map_filepath)
        base_map_np = kiva_env_str2number(base_map_str)

    elif domain == "competition":
        block_idxs = [
            competition_obj_types.index("@"),
        ]
        base_map_str, map_name = read_in_competition_map(map_filepath)
        base_map_np = competition_env_str2number(base_map_str)

    # Competition has the same order of direction
    right, up, left, down = kiva_directions

    h, w = base_map_np.shape

    edge_weights_matrix = np.ones((h, w, 4))

    for x in range(h):
        if x % 2 == 0:
            desired = right
        else:
            desired = left
        for y in range(w):
            # node_idx = w * x + y
            edge_weights_matrix[x][y][kiva_dir_map[desired]] = 0.5

    for y in range(w):
        if y % 2 == 0:
            desired = up
        else:
            desired = down
        for x in range(h):
            # node_idx = w * x + y
            edge_weights_matrix[x][y][kiva_dir_map[desired]] = 0.5

    wait_costs_matrix = np.ones((h, w))
    n_valid_vertices = get_n_valid_vertices(base_map_np, domain)

    if domain == "kiva":
        edge_weights_compress = kiva_compress_edge_weights(
            base_map_np, edge_weights_matrix, block_idxs)
        wait_cost_compress = [1.0]
    elif domain == "competition":
        comp_map = Map(map_filepath)
        edge_weights_compress = comp_compress_edge_matrix(
            comp_map, edge_weights_matrix.flatten())
        wait_cost_compress = comp_compress_vertex_matrix(
            comp_map, wait_costs_matrix.flatten())
    raw_weights = [
        *wait_cost_compress,
        *edge_weights_compress,
    ]

    # Name of the map
    graph_directedness = "bi-directed" if bi_directed else "directed"
    save_map_name = f"{map_name}_alternate_baseline_{graph_directedness}"

    # plot_highway_suggest_graph(
    #     base_map_np,
    #     raw_weights,
    #     save_map_name,
    #     store_dir=store_dir,
    # )
    plot_highway_edge_weights(
        base_map_np,
        raw_weights,
        save_map_name,
        store_dir=store_dir,
        domain=domain,
        map_filepath=map_filepath,
    )

    # For uni-directed graph, replace weights s.t.
    # 1 --> -1, invalid edges becomes -1
    # 0.5 --> 1, valid edges becomes 1
    if not bi_directed:
        for i in range(len(raw_weights)):
            if raw_weights[i] == 1:
                raw_weights[i] = -1
            else:
                raw_weights[i] = 1
        graph_directedness = "directed"
    write_map_str_to_json(
        os.path.join(store_dir, f"{save_map_name}.json"),
        base_map_str,
        save_map_name,
        domain,
        weight=True,
        weights=raw_weights,
        optimize_wait=True if domain=="competition" else False,
    )


if __name__ == "__main__":
    fire.Fire(gen_crisscross_baseline)
