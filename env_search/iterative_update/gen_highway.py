import os
import gin
import fire
import json

import multiprocessing

import numpy as np

from logdir import LogDir
from itertools import repeat
from env_search.competition.config import CompetitionConfig
from env_search.utils import (
    read_in_competition_map,
    read_in_kiva_map,
    get_n_valid_edges,
    get_n_valid_vertices,
    competition_env_str2number,
    kiva_env_str2number,
    write_map_str_to_json,
)
from env_search.iterative_update.algo import iterative_update_cmd
from env_search.utils import min_max_normalize


def gen_highway_single(
    logdir: LogDir,
    base_map_np,
    base_map_str,
    model_json,
    piu_config_file,
    num_agents,
    bi_directed=True,
    domain="competition",
    map_filepath=None,
):
    """Generate highway (edge weights and wait costs) given an update model.
    """
    n_valid_edges = get_n_valid_edges(base_map_np, bi_directed, domain)
    n_valid_vertices = get_n_valid_vertices(base_map_np, domain)

    result, weights = iterative_update_cmd(
        n_valid_edges=n_valid_edges,
        n_valid_vertices=n_valid_vertices,
        config_file=piu_config_file,
        model_params=np.array(model_json["params"]),
        num_agents=int(num_agents),
        domain=domain,
        kiva_map_filepath=map_filepath,
    )
    wait_costs = weights[:n_valid_vertices]
    edge_weights = weights[n_valid_vertices:]

    # normalize
    gin.parse_config_file(piu_config_file)
    config = CompetitionConfig()
    lb, ub = config.bounds
    wait_costs = min_max_normalize(wait_costs, lb, ub)
    edge_weights = min_max_normalize(edge_weights, lb, ub)
    weights_norm = np.concatenate([wait_costs, edge_weights]).tolist()

    # Write result
    map_name = f"n_agents={num_agents}.json"
    opt_highway_path = logdir.file(os.path.join("maps", map_name))
    write_map_str_to_json(
        opt_highway_path,
        base_map_str,
        map_name,
        domain,
        weight=True,
        weights=weights_norm,
        optimize_wait=True,
        piu_n_agent=int(num_agents),
    )


def gen_highway(
    n_agent_start,
    n_agent_end,
    n_agent_step,
    model_filepath,
    map_filepath,
    piu_config_file,
    bi_directed=True,
    n_workers=16,
    domain="competition",
    reload_dir=None,
):
    # Read in model
    with open(model_filepath, "r") as f:
        model_json = json.load(f)

    # Read in map
    if domain == "competition":
        base_map_str, base_map_name = read_in_competition_map(map_filepath)
        base_map_np = competition_env_str2number(base_map_str)
    elif domain == "kiva":
        base_map_str, base_map_name = read_in_kiva_map(map_filepath)
        base_map_np = kiva_env_str2number(base_map_str)

    # Create logdir
    name = f"piu_maps_{base_map_name}"
    if reload_dir is None:
        logdir = LogDir(name, rootdir="./logs", uuid=8)
        num_agents = np.arange(n_agent_start, n_agent_end, n_agent_step)
        n_tasks = len(num_agents)
    else:
        logdir = LogDir(name, custom_dir=reload_dir)
        map_dir = logdir.dir("maps")
        if os.path.isdir(logdir.dir("maps")):
            num_agents = []
            n_tasks = 0
            for num_agent in range(n_agent_start, n_agent_end, n_agent_step):
                map_file = os.path.join(map_dir, f"n_agents={num_agent}.json")
                if not os.path.isfile(map_file):
                    num_agents.append(num_agent)
                    n_tasks += 1
        else:
            num_agents = np.arange(n_agent_start, n_agent_end, n_agent_step)
            n_tasks = len(num_agents)

    pool = multiprocessing.Pool(n_workers)
    results_and_weights = pool.starmap(
        gen_highway_single,
        zip(
            repeat(logdir, n_tasks),
            repeat(base_map_np, n_tasks),
            repeat(base_map_str, n_tasks),
            repeat(model_json, n_tasks),
            repeat(piu_config_file, n_tasks),
            num_agents,
            repeat(bi_directed, n_tasks),
            repeat(domain, n_tasks),
            repeat(map_filepath, n_tasks),
        ),
    )


if __name__ == "__main__":
    fire.Fire(gen_highway)
