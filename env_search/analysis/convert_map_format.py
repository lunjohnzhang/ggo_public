import os
import json
import numpy as np
import pandas as pd
import fire
from env_search import G_GRAPH_OUT_DIR
from env_search.analysis.visualize_highway import convert_json_weights_to_matrix
from env_search.utils import (get_n_valid_edges, get_n_valid_vertices,
                              kiva_env_str2number, competition_env_str2number)


def write_json_map_to_csv(map_filepath, domain, store_dir=G_GRAPH_OUT_DIR):
    wait_costs_matrix, edge_weights_matrix = convert_json_weights_to_matrix(
        map_filepath, domain, fill_value=np.inf)
    # order of edge weights is right, up, left, down
    with open(map_filepath, "r") as f:
        map_json = json.load(f)
    map_str_list = map_json["layout"]
    name = map_json["name"]
    h, w = len(map_str_list), len(map_str_list[0])

    if domain == "kiva":
        map_np = kiva_env_str2number(map_str_list)
    elif domain == "competition":
        map_np = competition_env_str2number(map_str_list)

    n_valid_vertices = get_n_valid_vertices(map_np, domain)
    n_valid_edges = get_n_valid_edges(map_np, bi_directed=True, domain=domain)
    print(f"Valid vertices: {n_valid_vertices}, Valid edges: {n_valid_edges}")

    g_graph = {
        "id": [],
        "type": [],
        "x": [],
        "y": [],
        "weight_to_RIGHT": [],
        "weight_to_UP": [],
        "weight_to_LEFT": [],
        "weight_to_DOWN": [],
        "weight_for_WAIT": [],
    }

    for x in range(h):
        for y in range(w):
            g_graph["id"].append(x * w + y)
            g_graph["type"].append(map_str_list[x][y])
            g_graph["x"].append(x)
            g_graph["y"].append(y)
            g_graph["weight_to_RIGHT"].append(edge_weights_matrix[x, y, 0])
            g_graph["weight_to_UP"].append(edge_weights_matrix[x, y, 1])
            g_graph["weight_to_LEFT"].append(edge_weights_matrix[x, y, 2])
            g_graph["weight_to_DOWN"].append(edge_weights_matrix[x, y, 3])
            g_graph["weight_for_WAIT"].append(wait_costs_matrix[x, y])
    pd.DataFrame(g_graph).to_csv(
        os.path.join(store_dir, f"{name}_g_graph.csv"),
        index=False,
    )


if __name__ == "__main__":
    fire.Fire(write_json_map_to_csv)
