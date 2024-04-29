import os
import json
import gin
import fire
import numpy as np
import py_driver  # type: ignore # ignore pylance warning
from typing import Callable
from env_search.utils import kiva_env_str2number
from env_search.warehouse.config import WarehouseConfig
from env_search.competition.config import CompetitionConfig
from env_search.competition.module import CompetitionModule
from env_search.warehouse.module import WarehouseModule


def iterative_update_cmd(
    n_valid_edges,  # 2540 for comp, 
    n_valid_vertices,  # 819 for comp, 
    config_file,
    seed=0,
    init_weight_file=None,
    output=".",
    model_params=None,
    domain="competition",
    kiva_map_filepath=None,
    num_agents=None,
):
    gin.parse_config_file(config_file)
    if domain == "competition":
        config = CompetitionConfig()
        # overwrite num agents
        if num_agents is not None and num_agents != "":
            config.num_agents = num_agents
        module = CompetitionModule(config)
        return module.evaluate_iterative_update(
            model_params=model_params
            if model_params is not None else np.random.rand(4271),
            eval_logdir=output,
            n_valid_edges=n_valid_edges,
            n_valid_vertices=n_valid_vertices,
            seed=seed,
        )
    elif domain == "kiva":
        config = WarehouseConfig()
        module = WarehouseModule(config)
        with open(kiva_map_filepath, "r") as f:
            map_json = json.load(f)
            map_np = kiva_env_str2number(map_json["layout"])
        return module.evaluate_iterative_update(
            map_np=map_np,
            map_json=map_json,
            num_agents=num_agents,
            model_params=model_params
            if model_params is not None else np.random.rand(4271),
            eval_logdir=output,
            n_valid_edges=n_valid_edges,
            n_valid_vertices=n_valid_vertices,
            seed=seed,
        )


if __name__ == "__main__":
    fire.Fire(iterative_update_cmd)
