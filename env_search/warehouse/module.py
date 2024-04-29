"""WarehouseConfig and WarehouseModule.

Usage:
    # Run as a script to demo the WarehouseModule.
    python env_search/warehouse/module.py
"""

import os
import gin
import copy
import json
import time
import fire
import logging
import pathlib
import warnings
import warehouse_sim  # type: ignore # ignore pylance warning
import numpy as np
import shutil
import multiprocessing

from scipy.stats import entropy
from pprint import pprint
from typing import List
from dataclasses import dataclass
from itertools import repeat, product
from typing import Collection, Optional
from queue import Queue
from typing import Collection
from env_search import LOG_DIR
from env_search.utils.logging import setup_logging
from env_search.iterative_update import WarehouseIterUpdateEnv
from env_search.warehouse.config import WarehouseConfig
from env_search.warehouse.update_model import WarehouseBaseUpdateModel
from env_search.warehouse.warehouse_result import (WarehouseResult,
                                                   WarehouseMetadata)
from env_search.utils import (kiva_obj_types, MIN_SCORE, kiva_env_number2str,
                              kiva_env_str2number, format_env_str,
                              read_in_kiva_map, flip_tiles,
                              kiva_uncompress_edge_weights,
                              kiva_uncompress_wait_costs)

logger = logging.getLogger(__name__)


class WarehouseModule:

    def __init__(self, config: WarehouseConfig):
        self.config = config

    def evaluate(
        self,
        map_json: str,
        eval_logdir: pathlib.Path,
        sim_seed: int,
        agentNum: int,
        map_id: int,
        eval_id: int,
    ):
        """
        Repair map and run simulation

        Args:
            map (np.ndarray): input map in integer format
            parent_map (np.ndarray): parent solution of the map. Will be None if
                                     current sol is the initial population.
            eval_logdir (str): log dir of simulation
            n_evals (int): number of evaluations
            sim_seed (int): random seed for simulation. Should be different for
                            each solution
            repair_seed (int): random seed for repairing. Should be the same as
                               master seed
            w_mode (bool): whether to run with w_mode, which replace 'r' with
                           'w' in generated map layouts, where 'w' is a
                           workstation. Under w_mode, robots will start from
                           endpoints and their tasks will alternate between
                           endpoints and workstations.
            n_endpt (int): number of endpoint around each obstacle
            min_n_shelf (int): min number of shelves
            max_n_shelf (int): max number of shelves
            agentNum (int): number of drives
            map_id (int): id of the current map to be evaluated. The id
                          is only unique to each batch, NOT to the all the
                          solutions. The id can make sure that each simulation
                          gets a different log directory.
        """
        output = str(eval_logdir / f"id_{map_id}-sim_{eval_id}-seed={sim_seed}")

        # We need to construct kwargs manually because some parameters
        # must NOT be passed in in order to use the default values
        # defined on the C++ side.
        # It is very dumb but works.

        kwargs = {
            "map": map_json,
            "output": output,
            "scenario": self.config.scenario,
            "task": self.config.task,
            "agentNum": agentNum,
            "cutoffTime": self.config.cutoffTime,
            "seed": int(sim_seed),
            "screen": self.config.screen,
            "solver": self.config.solver,
            "id": self.config.id,
            "single_agent_solver": self.config.single_agent_solver,
            "lazyP": self.config.lazyP,
            "simulation_time": self.config.simulation_time,
            "simulation_window": self.config.simulation_window,
            "travel_time_window": self.config.travel_time_window,
            "potential_function": self.config.potential_function,
            "potential_threshold": self.config.potential_threshold,
            "rotation": self.config.rotation,
            "robust": self.config.robust,
            "CAT": self.config.CAT,
            "hold_endpoints": self.config.hold_endpoints,
            "dummy_paths": self.config.dummy_paths,
            "prioritize_start": self.config.prioritize_start,
            "suboptimal_bound": self.config.suboptimal_bound,
            "log": self.config.log,
            "test": self.config.test,
            "force_new_logdir": True,
            "save_result": self.config.save_result,
            "save_solver": self.config.save_solver,
            "save_heuristics_table": self.config.save_heuristics_table,
            "stop_at_traffic_jam": self.config.stop_at_traffic_jam,
            "left_w_weight": self.config.left_w_weight,
            "right_w_weight": self.config.right_w_weight,
        }

        # For some of the parameters, we do not want to pass them in here
        # to the use the default value defined on the C++ side.
        # We are not able to define the default value in python because values
        # such as INT_MAX can be tricky in python but easy in C++.
        planning_window = self.config.planning_window
        if planning_window is not None:
            kwargs["planning_window"] = planning_window

        one_sim_result_jsonstr = warehouse_sim.run(**kwargs)

        result_json = json.loads(one_sim_result_jsonstr)
        return result_json

    def evaluate_iterative_update(
        self,
        map_np: np.ndarray,
        map_json: str,
        num_agents: int,
        model_params: List,
        eval_logdir: str,
        n_valid_edges: int,
        n_valid_vertices: int,
        seed: int,
    ):
        iter_update_env = WarehouseIterUpdateEnv(
            map_np,
            map_json,
            num_agents,
            eval_logdir,
            n_valid_vertices=n_valid_vertices,
            n_valid_edges=n_valid_edges,
            config=self.config,
            seed=seed,
            # init_weight_file=init_weight_file,
        )
        update_model: WarehouseBaseUpdateModel = \
            self.config.iter_update_model_type(
                map_np,
                model_params,
                n_valid_vertices,
                n_valid_edges,
            )
        all_throughputs = []
        obs, info = iter_update_env.reset()
        curr_wait_costs = info["curr_wait_costs"]
        curr_edge_weights = info["curr_edge_weights"]
        curr_result = info["result"]
        curr_throughput = curr_result["throughput"]
        all_throughputs.append(curr_throughput)
        done = False
        block_idx = [
            kiva_obj_types.index("@"),
        ]
        while not done:

            edge_usage_matrix = np.array(curr_result["edge_usage_matrix"])
            wait_usage_matrix = np.array(curr_result["vertex_wait_matrix"])

            # Get update value
            if not self.config.optimize_wait:
                curr_wait_costs_real = np.zeros(n_valid_vertices)
                curr_wait_costs_real[:] = curr_wait_costs
                curr_wait_costs = curr_wait_costs_real
            wait_cost_update_vals, edge_weight_update_vals = \
                update_model.get_update_values(
                    wait_usage_matrix,
                    edge_usage_matrix,
                    np.array(kiva_uncompress_wait_costs(map_np, curr_wait_costs, block_idx, fill_value=0)),
                    np.array(kiva_uncompress_edge_weights(map_np, curr_edge_weights, block_idx, fill_value=0)),
                )

            # Perform update
            if not self.config.optimize_wait:
                wait_cost_update_vals = np.mean(wait_cost_update_vals,
                                                keepdims=True)
            obs, imp_throughput, done, _, info = iter_update_env.step(
                np.concatenate([wait_cost_update_vals,
                                edge_weight_update_vals]))
            curr_wait_costs = info["curr_wait_costs"]
            curr_edge_weights = info["curr_edge_weights"]
            curr_throughput += imp_throughput
            all_throughputs.append(curr_throughput)
            curr_result = info["result"]

        # print(all_throughputs)
        # print(np.max(all_throughputs))
        if self.config.optimize_wait:
            return curr_result, np.concatenate(
                [curr_wait_costs, curr_edge_weights])
        else:
            return curr_result, np.concatenate([[curr_wait_costs],
                                                curr_edge_weights])

    def process_eval_result(
        self,
        edge_weights,
        wait_costs,
        curr_result_json: List[dict],
        n_evals: int,
        map_np_unrepaired: np.ndarray,
        map_comp_unrepaired: np.ndarray,
        map_np_repaired: np.ndarray,
        w_mode: bool,
        map_id: int,
    ):
        """
        Process the evaluation result

        Args:
            curr_result_json (List[dict]): result json of all simulations of 1
                map.

        """

        # Should never fail.
        # TODO: Detect any possible failure case.
        # # Deal with failed layout.
        # # For now, failure only happens during MILP repair, so if failure
        # # happens, all simulation json results would contain
        # # {"success": False}.
        # if not curr_result_json[0]["success"]:
        #     logger.info(
        #     f"Map ID {map_id} failed.")

        #     metadata = WarehouseMetadata(
        #         map_int_unrepaired=map_comp_unrepaired,
        #         map_int_raw=map_np_unrepaired,
        #     )
        #     result = WarehouseResult.from_raw(
        #         warehouse_metadata=metadata,
        #         opts={
        #             "aggregation": self.config.aggregation_type,
        #             "measure_names": self.config.measure_names,
        #         },
        #     )
        #     result.failed = True
        #     return result

        # Collect the results
        keys = curr_result_json[0].keys()
        collected_results = {key: [] for key in keys}
        for result_json in curr_result_json:
            for key in keys:
                collected_results[key].append(result_json[key])

        # Post process result if necessary
        tile_usage = np.array(collected_results.get("tile_usage"))
        edge_pair_usage = np.array(collected_results.get("edge_pair_usage"))
        # tile_usage = tile_usage.reshape(n_evals, *map_np_repaired.shape)
        tasks_finished_timestep = [
            np.array(x)
            for x in collected_results.get("tasks_finished_timestep")
        ]

        # Get objective based on type
        objs = None
        throughput = np.array(collected_results.get("throughput"))
        if self.config.obj_type == "throughput":
            objs = throughput
        else:
            return ValueError(
                f"Object type {self.config.obj_type} not supported")

        # Longest common subpath
        subpaths = collected_results.get("longest_common_path")
        subpath_len_mean = np.mean([len(path) for path in subpaths])

        # Create WarehouseResult object using the mean of n_eval simulations
        # For tile_usage, num_wait, and finished_task_len, the mean is not taken
        metadata = WarehouseMetadata(
            objs=objs,
            throughput=collected_results.get("throughput"),
            tile_usage=tile_usage,
            tile_usage_mean=np.mean(collected_results.get("tile_usage_mean")),
            tile_usage_std=np.mean(collected_results.get("tile_usage_std")),
            edge_weights=edge_weights,
            edge_weight_mean=np.mean(edge_weights),
            edge_weight_std=np.std(edge_weights),
            edge_pair_usage=edge_pair_usage,
            edge_pair_usage_mean=np.mean(
                collected_results.get("edge_pair_usage_mean")),
            edge_pair_usage_std=np.mean(
                collected_results.get("edge_pair_usage_std")),
            wait_costs=wait_costs,
            num_wait=collected_results.get("num_wait"),
            num_wait_mean=np.mean(collected_results.get("num_wait_mean")),
            num_wait_std=np.mean(collected_results.get("num_wait_std")),
            num_turns=collected_results.get("num_turns"),
            num_turns_mean=np.mean(collected_results.get("num_turns_mean")),
            num_turns_std=np.mean(collected_results.get("num_turns_std")),
            finished_task_len=collected_results.get("finished_task_len"),
            finished_len_mean=np.mean(
                collected_results.get("finished_len_mean")),
            finished_len_std=np.mean(collected_results.get("finished_len_std")),
            tasks_finished_timestep=tasks_finished_timestep,
            num_rev_action=collected_results.get("num_rev_action"),
            num_rev_action_mean=np.mean(
                collected_results.get("num_rev_action_mean")),
            num_rev_action_std=np.mean(
                collected_results.get("num_rev_action_std")),
            subpath=collected_results.get("subpath"),
            subpath_len_mean=subpath_len_mean,
        )
        result = WarehouseResult.from_raw(
            warehouse_metadata=metadata,
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


logger = logging.getLogger(__name__)
d = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def single_simulation(seed, agent_num, kwargs, results_dir):
    # seed += 50
    kwargs["seed"] = int(seed)
    output_dir = os.path.join(results_dir,
                              f"sim-agent_num={agent_num}-seed={seed}")
    kwargs["output"] = output_dir
    kwargs["agentNum"] = agent_num

    # Write kwargs to logdir
    os.mkdir(output_dir)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(json.dumps(kwargs, indent=4))

    result_jsonstr = warehouse_sim.run(**kwargs)
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
    warehouse_config,
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

    gin.parse_config_file(warehouse_config)

    # Read in map
    with open(map_filepath, "r") as f:
        raw_env_json = json.load(f)

    # Create log dir
    map_name = raw_env_json["name"]
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    base_log_dir = time_str + "_" + map_name
    log_dir = os.path.join(LOG_DIR, base_log_dir)
    results_dir = os.path.join(log_dir, "results")
    os.mkdir(log_dir)
    os.mkdir(results_dir)

    # Write map file to logdir
    with open(os.path.join(log_dir, "map.json"), "w") as f:
        f.write(json.dumps(raw_env_json, indent=4))

    # Construct kwargs
    kwargs = {
        "map":
            json.dumps(raw_env_json),
        # "output" : log_dir,
        "scenario":
            gin.query_parameter("WarehouseConfig.scenario"),
        "task":
            gin.query_parameter("WarehouseConfig.task"),
        "agentNum":
            agent_num,
        "cutoffTime":
            gin.query_parameter("WarehouseConfig.cutoffTime"),
        # "seed" : seed,
        "screen":
            gin.query_parameter("WarehouseConfig.screen"),
        "solver":
            gin.query_parameter("WarehouseConfig.solver"),
        "id":
            gin.query_parameter("WarehouseConfig.id"),
        "single_agent_solver":
            gin.query_parameter("WarehouseConfig.single_agent_solver"),
        "lazyP":
            gin.query_parameter("WarehouseConfig.lazyP"),
        "simulation_time":
            gin.query_parameter("WarehouseConfig.simulation_time"),
        "simulation_window":
            gin.query_parameter("WarehouseConfig.simulation_window"),
        "travel_time_window":
            gin.query_parameter("WarehouseConfig.travel_time_window"),
        "potential_function":
            gin.query_parameter("WarehouseConfig.potential_function"),
        "potential_threshold":
            gin.query_parameter("WarehouseConfig.potential_threshold"),
        "rotation":
            gin.query_parameter("WarehouseConfig.rotation"),
        "robust":
            gin.query_parameter("WarehouseConfig.robust"),
        "CAT":
            gin.query_parameter("WarehouseConfig.CAT"),
        "hold_endpoints":
            gin.query_parameter("WarehouseConfig.hold_endpoints"),
        "dummy_paths":
            gin.query_parameter("WarehouseConfig.dummy_paths"),
        "prioritize_start":
            gin.query_parameter("WarehouseConfig.prioritize_start"),
        "suboptimal_bound":
            gin.query_parameter("WarehouseConfig.suboptimal_bound"),
        "log":
            gin.query_parameter("WarehouseConfig.log"),
        "test":
            gin.query_parameter("WarehouseConfig.test"),
        "force_new_logdir":
            False,
        "save_result":
            gin.query_parameter("WarehouseConfig.save_result"),
        "save_solver":
            gin.query_parameter("WarehouseConfig.save_solver"),
        "save_heuristics_table":
            gin.query_parameter("WarehouseConfig.save_heuristics_table"),
        "stop_at_traffic_jam":
            gin.query_parameter("WarehouseConfig.stop_at_traffic_jam"),
        "left_w_weight":
            gin.query_parameter("WarehouseConfig.left_w_weight"),
        "right_w_weight":
            gin.query_parameter("WarehouseConfig.right_w_weight"),
    }

    # For some of the parameters, we do not want to pass them in here
    # to the use the default value defined on the C++ side.
    try:
        planning_window = gin.query_parameter("WarehouseConfig.planning_window")
        if planning_window is not None:
            kwargs["planning_window"] = planning_window
    except ValueError:
        pass

    have_run = set()
    if reload is not None and reload != "":
        all_results_dir = os.path.join(reload, "results")
        for result_dir in os.listdir(all_results_dir):
            result_dir_full = os.path.join(all_results_dir, result_dir)
            if os.path.exists(os.path.join(result_dir_full, "result.json")) and\
               os.path.exists(os.path.join(result_dir_full, "config.json")):
                curr_configs = result_dir.split("-")
                curr_agent_num = int(curr_configs[1].split("=")[1])
                curr_seed = int(curr_configs[2].split("=")[1])
                have_run.add((curr_agent_num, curr_seed))
            else:
                breakpoint()
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
