import os
import fire

from env_search.utils import (
    get_n_valid_vertices,
    get_n_valid_edges,
    read_in_competition_map,
    read_in_kiva_map,
    competition_env_str2number,
    kiva_env_str2number,
)

map_list = [
    ("maps/competition/human/pibt_random_unweight_32x32.json", "competition"),
    ("maps/competition/human/pibt_room-64-64-8.json", "competition"),
    ("maps/warehouse/human/kiva_large_w_mode.json", "kiva"),
    ("maps/warehouse/human/kiva_small_r_mode.json", "kiva"),
]


def map_metrics():
    for map_filepath, domain in map_list:
        if domain == "kiva":
            map_str, name = read_in_kiva_map(map_filepath)
            map_np = kiva_env_str2number(map_str)
        elif domain == "competition":
            map_str, name = read_in_competition_map(map_filepath)
            map_np = competition_env_str2number(map_str)

        n_valid_edges = get_n_valid_edges(map_np,
                                          bi_directed=True,
                                          domain=domain)
        n_valid_vertices = get_n_valid_vertices(map_np, domain)

        print(f"{name} ({n_valid_vertices}, {n_valid_edges})")


if __name__ == "__main__":
    fire.Fire(map_metrics)
