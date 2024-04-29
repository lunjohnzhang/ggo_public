"""CompetitionConfig and CompetitionModule.

Usage:
    # Run as a script to demo the CompetitionModule.
    python env_search/competition/module.py
"""

import os
import gc
import gin
import copy
import json
import time
import fire
import logging
import pathlib
import warnings
import py_driver  # type: ignore # ignore pylance warning
import numpy as np
import shutil
import multiprocessing
import hashlib
import subprocess

from scipy.stats import entropy
from pprint import pprint
from dataclasses import dataclass
from itertools import repeat, product
from typing import Collection, Optional, Tuple, List, Callable
from queue import Queue
from env_search import LOG_DIR, COMPETITION_DIR
from env_search.utils.logging import setup_logging
from env_search.competition.config import CompetitionConfig
from env_search.competition.competition_result import (CompetitionResult,
                                                       CompetitionMetadata)
from env_search.utils import (MIN_SCORE, competition_obj_types,
                              competition_env_str2number, get_n_valid_edges,
                              get_n_valid_vertices, load_pibt_default_config,
                              single_sim_done, get_project_dir)
from env_search.utils.logging import get_current_time_str
from env_search.iterative_update import CompetitionIterUpdateEnv
from env_search.competition.update_model import CompetitionBaseUpdateModel, CompetitionCNNUpdateModel
from env_search.competition.update_model.utils import (
    Map, comp_uncompress_edge_matrix, comp_uncompress_vertex_matrix,
    comp_compress_edge_matrix, comp_compress_vertex_matrix)

logger = logging.getLogger(__name__)


class CompetitionModule:

    def __init__(self, config: CompetitionConfig):
        self.config = config

    def _run_sim_single(
        self,
        kwargs,
        manually_clean_memory=True,
        save_in_disk=True,
    ):
        if not manually_clean_memory:
            one_sim_result_jsonstr = py_driver.run(**kwargs)
            result_json = json.loads(one_sim_result_jsonstr)
            return result_json
        else:
            if save_in_disk:
                file_dir = os.path.join(get_project_dir(), 'run_files')
                os.makedirs(file_dir, exist_ok=True)
                hash_obj = hashlib.sha256()
                raw_name = get_current_time_str().encode() + os.urandom(16)
                hash_obj.update(raw_name)
                file_name = hash_obj.hexdigest()
                file_path = os.path.join(file_dir, file_name)
                with open(file_path, 'w') as f:
                    json.dump(kwargs, f)
                t1 = time.time()
                delimiter1 = "----DELIMITER1----DELIMITER1----"
                # delimiter2 = "----DELIMITER2----DELIMITER2----"
                output = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
import py_driver
import json
import time

file_path='{file_path}'
with open(file_path, 'r') as f:
    kwargs_ = json.load(f)

one_sim_result_jsonstr = py_driver.run(**kwargs_)
result_json = json.loads(one_sim_result_jsonstr)

print("{delimiter1}")
print(result_json)
print("{delimiter1}")

                """
                    ],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
                t2 = time.time()
                if os.path.exists(file_path):
                    os.remove(file_path)
                else:
                    raise NotImplementedError

            else:
                t1 = time.time()
                delimiter1 = "----DELIMITER1----DELIMITER1----"
                output = subprocess.run(
                    [
                        'python', '-c', f"""\
import numpy as np
import py_driver
import json

kwargs_ = {kwargs}
one_sim_result_jsonstr = py_driver.run(**kwargs_)
result_json = json.loads(one_sim_result_jsonstr)
np.set_printoptions(threshold=np.inf)
print("{delimiter1}")
print(result_json)
print("{delimiter1}")
                    """
                    ],
                    stdout=subprocess.PIPE).stdout.decode('utf-8')
                t2 = time.time()
            # print("================")
            # if self.verbose >= 2:
            #     print("run_sim time = ", t2-t1)

            # if self.verbose >= 4:
            #     o = output.split(delimiter2)
            #     for t in o[1:-1:2]:
            #         time_s = t.replace('\n', '')
            #         print("inner sim time =", time_s)
            #     print(self.config.iter_update_n_sim)
            outputs = output.split(delimiter1)
            if len(outputs) <= 2:
                print(output)
                raise NotImplementedError
            else:
                results_str = outputs[1].replace('\n', '').replace(
                    'array', 'np.array')
                # print(collected_results_str)
                results = eval(results_str)

            gc.collect()
            return results

    def evaluate(
        self,
        edge_weights_json: str,
        wait_costs_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        map_id: int,
        eval_id: int,
    ):
        """
        Run simulation

        Args:
            edge_weights_json (str): json string of the edge weights
            wait_costs_json (str): json string of the wait costs
            eval_logdir (str): log dir of simulation
            n_evals (int): number of evaluations
            sim_seed (int): random seed for simulation. Should be different for
                            each solution
            map_id (int): id of the current map to be evaluated. The id
                          is only unique to each batch, NOT to the all the
                          solutions. The id can make sure that each simulation
                          gets a different log directory.
            eval_id (int): id of evaluation.
        """
        # output = str(eval_logdir / f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        kwargs = {
            "map_json_path": self.config.map_path,
            "simulation_steps": self.config.simulation_time,
            "gen_random": self.config.gen_random,
            "num_tasks": self.config.num_tasks,
            "num_agents": self.config.num_agents,
            "weights": edge_weights_json,
            "wait_costs": wait_costs_json,
            "plan_time_limit": self.config.plan_time_limit,
            "seed": int(sim_seed),
            "preprocess_time_limit": self.config.preprocess_time_limit,
            "file_storage_path": self.config.file_storage_path,
            "task_assignment_strategy": self.config.task_assignment_strategy,
            "num_tasks_reveal": self.config.num_tasks_reveal,
            "config": load_pibt_default_config(),  # Use PIBT default config
        }

        result_json = self._run_sim_single(kwargs,
                                           manually_clean_memory=True,
                                           save_in_disk=True)
        return result_json

    def evaluate_iterative_update(
        self,
        model_params: List,
        eval_logdir: str,
        n_valid_edges: int,
        n_valid_vertices: int,
        seed: int,
    ):
        """Run PIU

        Args:
            model_params (List): parameters of the update model
            eval_logdir (str): log dir
            n_valid_edges (int): number of valid edges
            n_valid_vertices (int): number of valid vertices
            seed (int): random seed
        """
        iter_update_env = CompetitionIterUpdateEnv(
            n_valid_vertices=n_valid_vertices,
            n_valid_edges=n_valid_edges,
            config=self.config,
            seed=seed,
            # init_weight_file=init_weight_file,
        )
        comp_map = Map(self.config.map_path)
        update_mdl_kwargs = {}
        if self.config.iter_update_mdl_kwargs is not None:
            update_mdl_kwargs = self.config.iter_update_mdl_kwargs
        update_model: CompetitionBaseUpdateModel = \
            self.config.iter_update_model_type(
                comp_map,
                model_params,
                n_valid_vertices,
                n_valid_edges,
                **update_mdl_kwargs,
            )
        all_throughputs = []
        obs, info = iter_update_env.reset()
        # curr_wait_costs = info["curr_wait_costs"]
        # curr_edge_weights = info["curr_edge_weights"]
        curr_result = info["result"]
        curr_throughput = curr_result["throughput"]
        all_throughputs.append(curr_throughput)
        done = False
        while not done:
            edge_usage_matrix = np.moveaxis(obs[:4], 0, 2)
            wait_usage_matrix = obs[4]
            curr_edge_weights_matrix = np.moveaxis(obs[5:9], 0, 2)
            curr_wait_costs_matrix = obs[9]

            # Get update value
            wait_cost_update_vals, edge_weight_update_vals = \
                update_model.get_update_values(
                    wait_usage_matrix,
                    edge_usage_matrix,
                    curr_wait_costs_matrix,
                    curr_edge_weights_matrix,
                )

            # Perform update
            obs, imp_throughput, done, _, info = iter_update_env.step(
                np.concatenate([wait_cost_update_vals,
                                edge_weight_update_vals]))

            curr_throughput += imp_throughput
            all_throughputs.append(curr_throughput)
            curr_result = info["result"]

        # print(all_throughputs)
        # print(np.max(all_throughputs))
        # breakpoint()
        curr_wait_costs = info["curr_wait_costs"]
        curr_edge_weights = info["curr_edge_weights"]
        # curr_edge_weights = comp_compress_edge_matrix(
        #     comp_map, np.moveaxis(obs[5:9], 0, 2))
        # curr_wait_costs = comp_compress_vertex_matrix(obs[9])

        return curr_result, np.concatenate([curr_wait_costs, curr_edge_weights])

    def process_eval_result(
        self,
        edge_weights,
        wait_costs,
        curr_result_json: List[dict],
        n_evals: int,
        map_id: int,
    ):
        """
        Process the evaluation result

        Args:
            curr_result_json (List[dict]): result json of all simulations of 1
                map.

        """

        # Collect the results
        keys = curr_result_json[0].keys()
        collected_results = {key: [] for key in keys}
        for result_json in curr_result_json:
            for key in keys:
                collected_results[key].append(result_json[key])

        # Post process result if necessary
        tile_usage = np.array(collected_results.get("tile_usage"), dtype=float)
        tile_usage /= np.sum(tile_usage)
        # tile_usage = tile_usage.reshape(n_evals, *map_np_repaired.shape)
        tile_usage_mean = np.mean(tile_usage, axis=1)
        tile_usage_std = np.std(tile_usage, axis=1)
        edge_pair_usage = np.array(collected_results.get("edge_pair_usage"))
        edge_pair_usage_mean = collected_results.get("edge_pair_usage_mean")
        edge_pair_usage_std = collected_results.get("edge_pair_usage_std")

        logger.info(f"Mean tile-usage: {np.mean(tile_usage_mean)}")
        logger.info(f"Mean edge-pair-usage: {np.mean(edge_pair_usage_mean)}")
        logger.info(f"Std of wait cost: {np.std(wait_costs)}")

        # Get objective based on type
        objs = None
        throughput = np.array(collected_results.get("throughput"))
        if self.config.obj_type == "throughput":
            objs = throughput
        else:
            return ValueError(
                f"Object type {self.config.obj_type} not supported")
        logger.info(f"Throughputs: {throughput}")
        # Create CompetitionResult object using the mean of n_eval simulations
        # For tile_usage, num_wait, and finished_task_len, the mean is not taken
        metadata = CompetitionMetadata(
            objs=objs,
            throughput=collected_results.get("throughput"),
            tile_usage=tile_usage,
            tile_usage_mean=np.mean(tile_usage_mean),
            tile_usage_std=np.mean(tile_usage_std),
            edge_weight_mean=np.mean(edge_weights),
            edge_weight_std=np.std(edge_weights),
            edge_weights=edge_weights,
            edge_pair_usage=edge_pair_usage,
            edge_pair_usage_mean=np.mean(edge_pair_usage_mean),
            edge_pair_usage_std=np.mean(edge_pair_usage_std),
            wait_cost_mean=np.mean(wait_costs),
            wait_cost_std=np.std(wait_costs),
            wait_costs=wait_costs,
        )
        result = CompetitionResult.from_raw(
            competition_metadata=metadata,
            opts={
                "aggregation": self.config.aggregation_type,
                "measure_names": self.config.measure_names,
            },
        )

        return result

    def actual_qd_score(self, objs):
        """Calculates QD score of the given objectives.

        Scores are normalized to be non-negative by subtracting a constant min
        score.

        Args:
            objs: List of objective values.
        """
        objs = np.array(objs)
        objs -= MIN_SCORE
        if np.any(objs < 0):
            warnings.warn("Some objective values are still negative.")
        return np.sum(objs)


# logger = logging.getLogger(__name__)
d = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def single_simulation(seed, agent_num, kwargs, results_dir):
    kwargs["seed"] = int(seed)
    output_dir = os.path.join(results_dir,
                              f"sim-agent_num={agent_num}-seed={seed}")
    # kwargs["output"] = output_dir
    kwargs["num_agents"] = agent_num

    # Write kwargs to logdir
    os.mkdir(output_dir)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(json.dumps(kwargs, indent=4))

    result_jsonstr = py_driver.run(**kwargs)
    result_json = json.loads(result_jsonstr)

    throughput = result_json["throughput"]

    # Write result to logdir
    # Load and then dump to format the json
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(output_dir, f"result.json"), "w") as f:
        f.write(json.dumps(result_json, indent=4))

    return throughput


def main(
    competition_config,
    map_filepath,
    agent_num=10,
    agent_num_step_size=1,
    seed=0,
    n_evals=10,
    n_sim=2,  # Run `inc_agents` `n_sim` times
    mode="constant",
    n_workers=32,
    reload=None,
):
    """
    For testing purposes. Graph a map and run one simulation.

    Args:
        mode: "constant", "inc_agents", or "inc_timesteps".
              "constant" will run `n_eval` simulations with the same
              `agent_num`.
              "increase" will run `n_eval` simulations with an inc_agents
              number of `agent_num`.
    """
    setup_logging(on_worker=False)

    gin.parse_config_file(competition_config)

    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)

    # Create log dir
    map_name = raw_env_json["name"]
    map_np = competition_env_str2number(raw_env_json["layout"])
    optimize_wait = raw_env_json[
        "optimize_wait"] if "optimize_wait" in raw_env_json else False
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = time_str + "_" + map_name
    log_dir = os.path.join(LOG_DIR, base_log_dir)
    results_dir = os.path.join(log_dir, "results")
    os.mkdir(log_dir)
    os.mkdir(results_dir)

    # Write map file to logdir
    with open(os.path.join(log_dir, "map.json"), "w") as f:
        f.write(json.dumps(raw_env_json, indent=4))

    n_valid_vertices = get_n_valid_vertices(map_np, "competition")
    n_valid_edges = get_n_valid_edges(map_np, True, "competition")

    if optimize_wait:
        if "weights" in raw_env_json:
            all_weights = raw_env_json["weights"]
        else:
            all_weights = np.ones(n_valid_edges + n_valid_vertices).tolist()

        wait_costs = all_weights[:n_valid_vertices]
        edge_weights = all_weights[n_valid_vertices:]

        # kwargs = {
        #     "cmd": cmd,
        #     "weights": json.dumps(edge_weights),
        #     "wait_costs": json.dumps(wait_costs),
        # }

        kwargs = {
            "weights":
                json.dumps(edge_weights),
            "wait_costs":
                json.dumps(wait_costs),
            "map_json_path":
                map_filepath,
            "simulation_steps":
                gin.query_parameter("CompetitionConfig.simulation_time"),
            # Defaults
            "gen_random":
                True,
            "num_tasks":
                100000,
            "plan_time_limit":
                1,
            "preprocess_time_limit":
                1800,
            "file_storage_path":
                "large_files",
            "task_assignment_strategy":
                "roundrobin",
            "num_tasks_reveal":
                1,
            "config":
                load_pibt_default_config(),  # Use PIBT default config
        }
    else:
        if "weights" in raw_env_json:
            all_weights = raw_env_json["weights"]
        else:
            all_weights = np.ones(n_valid_edges + 1).tolist()
        kwargs = {
            "weights":
                json.dumps(all_weights),
            "map_json_path":
                map_filepath,
            "simulation_steps":
                gin.query_parameter("CompetitionConfig.simulation_time"),
            # Defaults
            "gen_random":
                True,
            "num_tasks":
                100000,
            "plan_time_limit":
                1,
            "preprocess_time_limit":
                1800,
            "file_storage_path":
                "large_files",
            "task_assignment_strategy":
                "roundrobin",
            "num_tasks_reveal":
                1,
            "config":
                load_pibt_default_config(),  # Use PIBT default config
        }
    have_run = set()
    if reload is not None and reload != "":
        all_results_dir = os.path.join(reload, "results")
        for result_dir in os.listdir(all_results_dir):
            result_dir_full = os.path.join(all_results_dir, result_dir)
            if single_sim_done(result_dir_full):
                # File exists, need to make sure it's not empty
                curr_configs = result_dir.split("-")
                curr_agent_num = int(curr_configs[1].split("=")[1])
                curr_seed = int(curr_configs[2].split("=")[1])
                have_run.add((curr_agent_num, curr_seed))
            else:
                # breakpoint()
                print(f"Removing incomplete logdir {result_dir_full}")
                shutil.rmtree(result_dir_full)

    pool = multiprocessing.Pool(n_workers)
    if mode == "inc_agents":
        seeds = []
        agent_nums = []
        agent_num_range = range(0, n_evals, agent_num_step_size)
        actual_n_evals = len(agent_num_range)
        for i in range(n_sim):
            for j in agent_num_range:
                curr_seed = seed + i
                curr_agent_num = agent_num + j
                if (curr_agent_num, curr_seed) in have_run:
                    continue
                seeds.append(curr_seed)
                agent_nums.append(curr_agent_num)
        throughputs = pool.starmap(
            single_simulation,
            zip(seeds, agent_nums,
                repeat(kwargs, actual_n_evals * n_sim - len(have_run)),
                repeat(results_dir, actual_n_evals * n_sim - len(have_run))),
        )
    elif mode == "constant":
        agent_nums = [agent_num for _ in range(n_evals)]
        seeds = np.random.choice(np.arange(10000), size=n_evals, replace=False)

        throughputs = pool.starmap(
            single_simulation,
            zip(seeds, agent_nums, repeat(kwargs, n_evals),
                repeat(results_dir, n_evals)),
        )

    avg_obj = np.mean(throughputs)
    max_obj = np.max(throughputs)
    min_obj = np.min(throughputs)

    n_evals = actual_n_evals if mode == "inc_agents" else n_evals

    print(f"Average throughput over {n_evals} simulations: {avg_obj}")
    print(f"Max throughput over {n_evals} simulations: {max_obj}")
    print(f"Min throughput over {n_evals} simulations: {min_obj}")


if __name__ == "__main__":
    fire.Fire(main)
