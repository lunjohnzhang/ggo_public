"""Provides CompetitionManager."""
import logging
from pathlib import Path
from typing import List, Tuple

import os
import time
import gin
import numpy as np
import copy
import json
from dask.distributed import Client
from logdir import LogDir
from tqdm import tqdm
from itertools import repeat

from cma.constraints_handler import BoundTransform

from env_search.device import DEVICE
from env_search.competition.emulation_model.buffer import Experience
from env_search.competition.emulation_model.aug_buffer import AugExperience
from env_search.competition.emulation_model.double_aug_buffer import DoubleAugExperience
from env_search.competition.emulation_model.emulation_model import CompetitionEmulationModel
from env_search.competition.emulation_model.networks import (
    CompetitionAugResnetOccupancy, CompetitionAugResnetRepairedMapAndOccupancy)
from env_search.competition.module import (CompetitionModule, CompetitionConfig)
from env_search.competition.run import (run_competition,
                                        process_competition_eval_result,
                                        run_competition_iterative_update)
from env_search.utils.worker_state import init_competition_module
from env_search.utils import (competition_obj_types, competition_env_number2str,
                              competition_env_str2number, format_env_str,
                              read_in_competition_map, get_n_valid_edges,
                              get_n_valid_vertices, min_max_normalize_2d)

logger = logging.getLogger(__name__)


@gin.configurable(denylist=["client", "rng"])
class CompetitionManager:
    """Manager for the competition environments.

    Args:
        client: Dask client for distributed compute.
        logdir: logdir of the experiment.
        rng: Random generator. Can be set later. Uses `np.random.default_rng()`
            by default.
        bound_handle (str):
            None: no bound handle
            projection: project out of bound solutions to bounds
            reflection: reflect out of bound solutions around bounds
            transformation: use `BoundTransform` from pycma to transform the
                solutions
            normalization: use min-max normalization to transform each solution
                to within the bound
        bounds (tuple): upper and lower bounds of solutions
        iterative_update (bool): if True, search for iterative update function
            parameters instead of edge weights/wait costs directly.
        update_model_n_params (int): number of parameters in the update model.
        n_evals: Number of times to evaluate each solution during real
            evaluation.
        lvl_width: Width of the map.
        lvl_height: Height of the map.
        bi_directed (bool): if true, the graph is bi-directed, otherwise the
                            graph is uni-directed.
        optimize_wait (bool): if true, optimize the cost of wait action.
    """

    def __init__(
        self,
        client: Client,
        logdir: LogDir,
        rng: np.random.Generator = None,
        bound_handle: str = None,
        bounds=None,
        iterative_update: bool = False,
        update_model_n_params: int = -1,
        n_evals: int = gin.REQUIRED,
        lvl_width: int = gin.REQUIRED,
        lvl_height: int = gin.REQUIRED,
        base_map_path: str = gin.REQUIRED,
        bi_directed: bool = gin.REQUIRED,
        optimize_wait: bool = gin.REQUIRED,
    ):
        self.client = client
        self.rng = rng or np.random.default_rng()

        self.n_evals = n_evals
        self.eval_batch_idx = 0  # index of each batch of evaluation

        self.logdir = logdir

        self.lvl_width = lvl_width
        self.lvl_height = lvl_height

        self.bi_directed = bi_directed
        self.optimize_wait = optimize_wait

        # Set up a module locally and on workers. During evaluations,
        # the util functions retrieves this module and uses it to
        # evaluate the function. Configuration is done with gin (i.e. the
        # params are in the config file).
        self.module = CompetitionModule(config := CompetitionConfig())
        client.register_worker_callbacks(
            lambda: init_competition_module(config))

        self.emulation_model = None

        # Read in map as both str and np
        self.base_map_str, _ = read_in_competition_map(base_map_path)
        self.base_map_np = competition_env_str2number(self.base_map_str)

        # bounds
        self.bounds = bounds
        self.bound_handle = bound_handle
        self.bound_transform = BoundTransform(list(
            self.bounds)) if self.bounds is not None else None

        # Runtime
        self.repair_runtime = 0
        self.sim_runtime = 0

        # Valid number of edges/vertices
        self.n_valid_edges = self.get_n_valid_edges()
        self.n_valid_vertices = self.get_n_valid_vertices()

        # Iterative update
        self.iterative_update = iterative_update
        self.update_model_n_params = update_model_n_params

    def em_init(self,
                seed: int,
                pickle_path: Path = None,
                pytorch_path: Path = None):
        """Initialize the emulation model and optionally load from saved state.

        Args:
            seed: Random seed to use.
            pickle_path: Path to the saved emulation model data (optional).
            pytorch_path: Path to the saved emulation model network (optional).
        """
        self.emulation_model = CompetitionEmulationModel(seed=seed + 420)
        if pickle_path is not None:
            self.emulation_model.load(pickle_path, pytorch_path)
        logger.info("Emulation Model: %s", self.emulation_model)

    def get_initial_sols(self, size: Tuple):
        """Returns random solutions with the given size.

        Args:
            size: Tuple with (n_solutions, sol_size).

        Returns:
            Randomly generated solutions.
        """
        raise NotImplementedError()

    def em_train(self):
        self.emulation_model.train()

    def emulation_pipeline(self, sols):
        """Pipeline that takes solutions and uses the emulation model to predict
        the objective and measures.

        Args:
            sols: Emitted solutions.

        Returns:
            lvls: Generated levels.
            objs: Predicted objective values.
            measures: Predicted measure values.
            success_mask: Array of size `len(lvls)`. An element in the array is
                False if some part of the prediction pipeline failed for the
                corresponding solution.
        """
        raise NotImplementedError()
        # n_maps = len(sols)
        # maps = np.array(sols).reshape(
        #     (n_maps, self.lvl_height, self.lvl_width)).astype(int)

        # success_mask = np.ones(len(maps), dtype=bool)
        # objs, measures = self.emulation_model.predict(maps)
        # return maps, objs, measures, success_mask

    def eval_pipeline(self, unrepaired_sols, parent_sols=None, batch_idx=None):
        """Pipeline that takes a solution and evaluates it.

        Args:
            sols: Emitted solution.
            parent_sols: Parent solution of sols.

        Returns:
            Results of the evaluation.
        """
        n_sols = len(unrepaired_sols)
        map_ids = np.arange(n_sols)

        if not self.iterative_update:
            # Perform repair and then evaluate the solutions if searching for
            # edge weights/wait cost directly

            lb, ub = self.bounds
            repaired_sols = copy.deepcopy(unrepaired_sols)
            # Handle bounds. All handling here are `Darwinian` approach, where
            # `repaired` solutions are not passed to `tell` method
            if self.bound_handle is not None:
                # Project values out of bounds to the bounds.
                if self.bound_handle == "projection":
                    repaired_sols = np.clip(unrepaired_sols, lb, ub)
                # Reflect values out of bounds around the bounds.
                elif self.bound_handle == "reflection":
                    l_oob = unrepaired_sols < lb
                    repaired_sols[l_oob] = 2 * lb - unrepaired_sols[l_oob]
                    r_oob = unrepaired_sols > ub
                    repaired_sols[r_oob] = 2 * ub - unrepaired_sols[r_oob]
                # Transform solutions into feasible region using pycma
                # `BoundTransform`
                elif self.bound_handle == "transformation":
                    repaired_sols = np.array([
                        self.bound_transform.repair(sol)
                        for sol in unrepaired_sols
                    ])
                # Normalize the solutions to within the bounds
                elif self.bound_handle == "normalization":
                    if self.optimize_wait:
                        # Split wait costs and edge weights and normalize
                        # separately
                        wait_costs = unrepaired_sols[:, :self.n_valid_vertices]
                        edge_weights = unrepaired_sols[:,
                                                       self.n_valid_vertices:]
                        wait_costs = min_max_normalize_2d(wait_costs, lb, ub)
                        edge_weights = min_max_normalize_2d(
                            edge_weights, lb, ub)
                        repaired_sols = np.concatenate(
                            [wait_costs, edge_weights], axis=1)
                    else:
                        repaired_sols = min_max_normalize_2d(
                            unrepaired_sols, lb, ub)
            # If we are searching edge weights of a uni-directed graph, we need
            # the `weights` from the original map to transform the generated
            # solutions to its bidirected counterpart (i.e. add -1 to indicate
            # the blocked edges)
            if not self.bi_directed:
                # raw_weights = self.base_map_json["weights"]
                # new_sols = []
                # for sol in repaired_sols:
                #     if self.optimize_wait:
                #         j = 1
                #         new_sol = [sol[0]]
                #         new_sol.extend(copy.deepcopy(raw_weights))
                #     else:
                #         j = 0
                #         new_sol = copy.deepcopy(raw_weights)
                #     for i in range(len(raw_weights)):
                #         # Edge is valid
                #         if raw_weights[i] != -1:
                #             idx = i + 1 if self.optimize_wait else i
                #             new_sol[idx] = sol[j]
                #             j += 1
                #     assert j == len(sol)
                #     new_sols.append(new_sol)
                # actual_sols = np.array(new_sols)
                raise NotImplementedError()
            else:
                actual_sols = repaired_sols

            # Make each solution evaluation have a different seed. Note that we
            # assign seeds to solutions rather than workers, which means that we
            # are agnostic to worker configuration.
            evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                                 size=len(actual_sols),
                                                 endpoint=True)

            # Based on number of simulations (n_evals), create maps and
            # corresponding variables to simulate
            edge_weight_jsons_sim = []
            wait_costs_jsons_sim = []
            maps_id_sim = []
            maps_eval_seed_sim = []
            eval_id_sim = []
            for map_id, eval_seed in zip(map_ids, evaluation_seeds):
                for j in range(self.n_evals):
                    curr_sol = actual_sols[map_id].tolist()
                    if self.optimize_wait:
                        # While optimizing waits, the first `n_valid_vertices`
                        # parameters are wait costs, the rests are edge weights.
                        wait_costs_json = json.dumps(
                            curr_sol[:self.n_valid_vertices])
                        edge_weight_json = json.dumps(
                            curr_sol[self.n_valid_vertices:])
                    else:
                        # Otherwise wait cost is the first parameter and is
                        # passed in along with the edge weights
                        wait_costs_json = None
                        edge_weight_json = json.dumps(curr_sol)
                    wait_costs_jsons_sim.append(wait_costs_json)
                    edge_weight_jsons_sim.append(edge_weight_json)
                    maps_id_sim.append(map_id)
                    maps_eval_seed_sim.append(eval_seed + j)
                    eval_id_sim.append(j)

            # Then, evaluate the maps
            if batch_idx is None:
                batch_idx = self.eval_batch_idx
            eval_logdir = self.logdir.pdir(
                f"evaluations/eval_batch_{batch_idx}")
            self.eval_batch_idx += 1
            sim_start_time = time.time()
            sim_futures = [
                self.client.submit(
                    run_competition,
                    edge_weights_json=edge_weights_json,
                    wait_costs_json=wait_costs_json,
                    eval_logdir=eval_logdir,
                    sim_seed=sim_seed,
                    map_id=map_id,
                    eval_id=eval_id,
                ) for (
                    edge_weights_json,
                    wait_costs_json,
                    sim_seed,
                    map_id,
                    eval_id,
                ) in zip(
                    edge_weight_jsons_sim,
                    wait_costs_jsons_sim,
                    maps_eval_seed_sim,
                    maps_id_sim,
                    eval_id_sim,
                )
            ]
            logger.info("Collecting evaluations")
            results_json = self.client.gather(sim_futures)
            self.sim_runtime += time.time() - sim_start_time

            results_json_sorted = []
            for i in range(n_sols):
                curr_eval_results = []
                for j in range(self.n_evals):
                    curr_eval_results.append(results_json[i * self.n_evals + j])
                results_json_sorted.append(curr_eval_results)

            logger.info("Processing eval results")

            process_futures = [
                self.client.submit(
                    process_competition_eval_result,
                    # Use `repaired_sols` instead of `actual_sols` because
                    # `actual_sols` contains placeholder `-1` for directed
                    # graphs.
                    edge_weights=repaired_sols[map_id][1:]
                    if not self.optimize_wait else
                    repaired_sols[map_id][self.n_valid_vertices:],
                    wait_costs=[repaired_sols[map_id][0]]
                    if not self.optimize_wait else
                    repaired_sols[map_id][:self.n_valid_vertices],
                    curr_result_json=curr_result_json,
                    n_evals=self.n_evals,
                    map_id=map_id,
                ) for (
                    curr_result_json,
                    map_id,
                ) in zip(
                    results_json_sorted,
                    map_ids,
                )
            ]
            results = self.client.gather(process_futures)
        else:
            assert self.optimize_wait
            # Otherwise, evaluate using iterative update func
            iter_update_sols = unrepaired_sols
            evaluation_seeds = self.rng.integers(np.iinfo(np.int32).max / 2,
                                                 size=len(iter_update_sols),
                                                 endpoint=True)
            eval_logdir = self.logdir.pdir(
                f"evaluations/eval_batch_{batch_idx}")
            sim_start_time = time.time()
            sim_futures = [
                self.client.submit(
                    run_competition_iterative_update,
                    n_valid_edges=self.n_valid_edges,
                    n_valid_vertices=self.n_valid_vertices,
                    eval_logdir=eval_logdir,
                    model_params=sol,
                    seed=seed,
                ) for sol, seed in zip(iter_update_sols, evaluation_seeds)
            ]
            logger.info("Collecting evaluations")
            results_and_weights = self.client.gather(sim_futures)
            self.sim_runtime += time.time() - sim_start_time
            results_json = []
            all_weights = []
            for i in range(n_sols):
                result_json, curr_all_weights = results_and_weights[i]
                results_json.append(result_json)
                all_weights.append(curr_all_weights)

            logger.info("Processing eval results")

            process_futures = [
                self.client.submit(
                    process_competition_eval_result,
                    edge_weights=all_weights[map_id][self.n_valid_vertices:],
                    wait_costs=all_weights[map_id][:self.n_valid_vertices],
                    curr_result_json=[curr_result_json],
                    n_evals=1,  # For n_eval should be fixed to 1
                    map_id=map_id,
                ) for (curr_result_json, map_id) in zip(results_json, map_ids)
            ]
            results = self.client.gather(process_futures)

        return results

    def add_experience(self, sol, result):
        """Add required experience to the emulation model based on the solution
        and the results.

        Args:
            sol: Emitted solution.
            result: Evaluation result.
        """
        obj = result.agg_obj
        meas = result.agg_measures
        input_lvl = result.competition_metadata["map_int_unrepaired"]
        repaired_lvl = result.competition_metadata["map_int"]

        if self.emulation_model.pre_network is not None:
            # Mean of tile usage over n_evals
            avg_tile_usage = np.mean(result.competition_metadata["tile_usage"],
                                     axis=0)
            if isinstance(self.emulation_model.pre_network,
                          CompetitionAugResnetOccupancy):
                self.emulation_model.add(
                    AugExperience(sol, input_lvl, obj, meas, avg_tile_usage))
            elif isinstance(self.emulation_model.pre_network,
                            CompetitionAugResnetRepairedMapAndOccupancy):
                self.emulation_model.add(
                    DoubleAugExperience(sol, input_lvl, obj, meas,
                                        avg_tile_usage, repaired_lvl))
        else:
            self.emulation_model.add(Experience(sol, input_lvl, obj, meas))

    @staticmethod
    def add_failed_info(sol, result) -> dict:
        """Returns a dict containing relevant information about failed levels.

        Args:
            sol: Emitted solution.
            result: Evaluation result.

        Returns:
            Dict with failed level information.
        """
        failed_level_info = {
            "solution":
                sol,
            "map_int_unrepaired":
                result.competition_metadata["map_int_unrepaired"],
            "log_message":
                result.log_message,
        }
        return failed_level_info

    def get_sol_size(self):
        """Get number of parameters to optimize.
        """
        if self.iterative_update:
            return self.update_model_n_params
        else:
            if self.optimize_wait:
                # All wait costs are optimized separately.
                return self.n_valid_edges + self.n_valid_vertices
            else:
                # All wait costs are optimized as one param.
                return self.n_valid_edges + 1

    def get_n_valid_edges(self):
        """Get number of valid edges in the given base map.
        Valid edges:
            1. does not go beyond the map.
            2. does not go from/to an obstacle.
        """

        return get_n_valid_edges(self.base_map_np, self.bi_directed,
                                 "competition")

    def get_n_valid_vertices(self):
        """Get number of valid vertices (aka non-obstacle vertices)
        """
        return get_n_valid_vertices(self.base_map_np, "competition")
