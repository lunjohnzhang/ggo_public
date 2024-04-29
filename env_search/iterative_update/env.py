import os
import gymnasium as gym
import json
import numpy as np
import copy
import py_driver  # type: ignore # ignore pylance warning
import warehouse_sim  # type: ignore # ignore pylance warning

# from abc import ABC
from gymnasium import spaces

from env_search.warehouse.config import WarehouseConfig
from env_search.competition.config import CompetitionConfig
from env_search.competition.update_model.utils import (
    Map,
    comp_uncompress_edge_matrix,
    comp_uncompress_vertex_matrix,
)
from env_search.utils import (
    kiva_obj_types,
    min_max_normalize,
    kiva_uncompress_edge_weights,
    kiva_uncompress_wait_costs,
    load_pibt_default_config,
    get_project_dir,
)
import gc

from env_search.utils.logging import get_current_time_str
import hashlib
import time
import subprocess


class IterUpdateEnvBase(gym.Env):
    """Iterative update env base."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        n_valid_vertices,
        n_valid_edges,
        max_iter=10,
        init_weight_file=None,
    ):
        super().__init__()
        self.i = 0  # timestep
        self.n_valid_vertices = n_valid_vertices
        self.n_valid_edges = n_valid_edges
        self.max_iter = max_iter
        self.init_weight_file = init_weight_file
        self.action_space = spaces.Box(low=-100,
                                       high=100,
                                       shape=(n_valid_edges +
                                              n_valid_vertices,))

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(n_valid_edges +
                                                   n_valid_vertices,))

    def step(self, action):
        return NotImplementedError()

    def reset(self, seed=None, options=None):
        # return observation, info
        if self.init_weight_file is None or self.init_weight_file == "":
            self.curr_edge_weights = np.ones(self.n_valid_edges)
            self.curr_wait_costs = np.ones(self.n_valid_vertices)
        else:
            with open(self.init_weight_file, "r") as f:
                map_json = json.load(f)
                all_weights = map_json["weights"]
                self.curr_edge_weights = np.array(
                    all_weights[self.n_valid_vertices:])
                self.curr_wait_costs = np.array(
                    all_weights[:self.n_valid_vertices])

        # Get baseline throughput
        self.i = 1  # We will run 1 simulation in reset
        init_result = self._run_sim(init_weight=True)
        self.init_throughput = init_result["throughput"]
        self.curr_throughput = init_result["throughput"]
        info = {
            "result": init_result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }
        return np.concatenate(
            [
                self.curr_wait_costs,
                self.curr_edge_weights,
            ],
            dtype=np.float32,
        ), info

    def render(self):
        return NotImplementedError()

    def close(self):
        return NotImplementedError()


class CompetitionIterUpdateEnv(IterUpdateEnvBase):

    def __init__(
        self,
        # input_file,
        n_valid_vertices,
        n_valid_edges,
        # max_iter=10,
        config: CompetitionConfig,
        seed=0,
        init_weight_file=None,
        # simulation_time=1000,
    ):
        super().__init__(
            n_valid_vertices=n_valid_vertices,
            n_valid_edges=n_valid_edges,
            max_iter=config.iter_update_max_iter,
            init_weight_file=init_weight_file,
        )
        # self.input_file = input_file
        self.config = config
        # self.simulation_time = simulation_time
        self.comp_map = Map(self.config.map_path)
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed

        # Use CNN observation
        h, w = self.comp_map.height, self.comp_map.width
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(10, h, w))

        if self.config.bounds is not None:
            self.lb, self.ub = self.config.bounds
        else:
            self.lb, self.ub = None, None

    def _gen_obs(self, result):
        edge_usage_matrix = np.array(result["edge_usage_matrix"])
        wait_usage_matrix = np.array(result["vertex_wait_matrix"])
        wait_cost_matrix = np.array(
            comp_uncompress_vertex_matrix(self.comp_map, self.curr_wait_costs))
        edge_weight_matrix = np.array(
            comp_uncompress_edge_matrix(self.comp_map, self.curr_edge_weights))

        # Normalize
        wait_usage_matrix = min_max_normalize(wait_usage_matrix, 0, 1)
        edge_usage_matrix = min_max_normalize(edge_usage_matrix, 0, 1)
        wait_cost_matrix = min_max_normalize(wait_cost_matrix, 0.1, 1)
        edge_weight_matrix = min_max_normalize(edge_weight_matrix, 0.1, 1)

        h, w = self.comp_map.height, self.comp_map.width
        edge_usage_matrix = edge_usage_matrix.reshape(h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(h, w, 1)
        edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)
        input = np.concatenate(
            [
                edge_usage_matrix,
                wait_usage_matrix,
                edge_weight_matrix,
                wait_cost_matrix,
            ],
            axis=2,
            dtype=np.float32,
        )
        input = np.moveaxis(input, 2, 0)
        return input

    def _run_sim(self,
                 init_weight=False,
                 manually_clean_memory=True,
                 save_in_disk=True):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """
        # cmd = f"./lifelong_comp --inputFile {self.input_file} --simulationTime {self.simulation_time} --planTimeLimit 1 --fileStoragePath large_files/"

        # Initial weights are assumed to be valid
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                           self.ub).tolist()

        results = []
        kwargs = {
            # "cmd": cmd,
            "map_json_path": self.config.map_path,
            "simulation_steps": self.config.simulation_time,
            "gen_random": self.config.gen_random,
            "num_tasks": self.config.num_tasks,
            "num_agents": self.config.num_agents,
            "weights": json.dumps(edge_weights),
            "wait_costs": json.dumps(wait_costs),
            "plan_time_limit": self.config.plan_time_limit,
            # "seed": int(self.rng.integers(100000)),
            "preprocess_time_limit": self.config.preprocess_time_limit,
            "file_storage_path": self.config.file_storage_path,
            "task_assignment_strategy": self.config.task_assignment_strategy,
            "num_tasks_reveal": self.config.num_tasks_reveal,
            "config": load_pibt_default_config(),  # Use PIBT default config
        }

        if not manually_clean_memory:
            results = []  # List[json]
            for _ in range(self.config.iter_update_n_sim):
                kwargs["seed"] = int(self.rng.integers(100000))
                result_jsonstr = py_driver.run(**kwargs)
                result = json.loads(result_jsonstr)
                results.append(result)
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
                delimiter2 = "----DELIMITER2----DELIMITER2----"
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

results = []

t0 = time.time()
rng = np.random.default_rng(seed={self.seed})
for _ in range({self.config.iter_update_n_sim}):
    kwargs_["seed"] = int(rng.integers(100000))
    t0 = time.time()
    result_jsonstr = py_driver.run(**kwargs_)
    t1 = time.time()
    print("{delimiter2}")
    print(t1-t0)
    print("{delimiter2}")
    result = json.loads(result_jsonstr)
    results.append(result)
np.set_printoptions(threshold=np.inf)

print("{delimiter1}")
print(results)
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
results = []
for _ in range({self.config.iter_update_n_sim}):
    kwargs_["seed"] = int({self.rng.integers(100000)})
    result_jsonstr = py_driver.run(**kwargs_)
    result = json.loads(result_jsonstr)
    results.append(result)
np.set_printoptions(threshold=np.inf)
print("{delimiter1}")
print(results)
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
        # aggregate results
        keys = results[0].keys()
        collected_results = {key: [] for key in keys}

        for result_json in results:
            for key in keys:
                collected_results[key].append(result_json[key])

        for _ in range(self.config.iter_update_n_sim):
            kwargs["seed"] = int(self.rng.integers(100000))
            result_jsonstr = py_driver.run(**kwargs)
            result = json.loads(result_jsonstr)
            results.append(result)

        # aggregate results
        keys = results[0].keys()
        collected_results = {key: [] for key in keys}
        for result_json in results:
            for key in keys:
                collected_results[key].append(result_json[key])

        for key in keys:
            collected_results[key] = np.mean(collected_results[key], axis=0)

        return collected_results

    def step(self, action):
        self.i += 1  # increment timestep

        # The environment is fully observable, so the observation is the
        # current edge weights/wait costs
        wait_cost_update_vals = action[:self.n_valid_vertices]
        edge_weight_update_vals = action[self.n_valid_vertices:]
        self.curr_wait_costs = wait_cost_update_vals
        self.curr_edge_weights = edge_weight_update_vals

        # Reward is difference between new throughput and current throughput
        result = self._run_sim()
        new_throughput = result["throughput"]
        reward = new_throughput - self.curr_throughput
        self.curr_throughput = new_throughput

        # terminated/truncate only if max iter is passed
        terminated = self.i >= self.max_iter
        truncated = terminated

        # Info includes the results
        info = {
            "result": result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        return self._gen_obs(result), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        init_result = info["result"]
        return self._gen_obs(init_result), info


class WarehouseIterUpdateEnv(IterUpdateEnvBase):

    def __init__(
        self,
        map_np,
        map_json,
        num_agents,
        eval_logdir,
        n_valid_vertices,
        n_valid_edges,
        config,
        seed=0,
        init_weight_file=None,
    ):
        super().__init__(
            n_valid_vertices=n_valid_vertices,
            n_valid_edges=n_valid_edges,
            max_iter=config.iter_update_max_iter,
            init_weight_file=init_weight_file,
        )
        self.config = config
        self.map_np = map_np
        self.map_json = map_json
        self.num_agents = num_agents
        self.eval_logdir = eval_logdir
        self.block_idxs = [
            kiva_obj_types.index("@"),
        ]
        self.rng = np.random.default_rng(seed=seed)

        # Use CNN observation
        h, w = self.map_np.shape
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(10, h, w))

        if self.config.bounds is not None:
            self.lb, self.ub = self.config.bounds
        else:
            self.lb, self.ub = None, None

    def _gen_obs(self, result):
        edge_usage_matrix = np.array(result["edge_usage_matrix"])
        wait_usage_matrix = np.array(result["vertex_wait_matrix"])

        edge_weight_matrix = np.array(
            kiva_uncompress_edge_weights(self.map_np,
                                         self.curr_edge_weights,
                                         self.block_idxs,
                                         fill_value=0))

        # While optimizing all wait costs, all entries of `wait_cost_matrix`
        # are different.
        if self.config.optimize_wait:
            wait_cost_matrix = np.array(
                kiva_uncompress_wait_costs(self.map_np,
                                           self.curr_wait_costs,
                                           self.block_idxs,
                                           fill_value=0))
        # Otherwise, `self.curr_wait_costs` is a single number and so all wait
        # costs are the same, but we need to transform it to a matrix.
        else:
            curr_wait_costs_compress = np.zeros(self.n_valid_vertices)
            curr_wait_costs_compress[:] = self.curr_wait_costs
            wait_cost_matrix = np.array(
                kiva_uncompress_wait_costs(self.map_np,
                                           curr_wait_costs_compress,
                                           self.block_idxs,
                                           fill_value=0))

        # Normalize
        wait_usage_matrix = min_max_normalize(wait_usage_matrix, 0, 1)
        edge_usage_matrix = min_max_normalize(edge_usage_matrix, 0, 1)
        wait_cost_matrix = min_max_normalize(wait_cost_matrix, 0.1, 1)
        edge_weight_matrix = min_max_normalize(edge_weight_matrix, 0.1, 1)

        h, w = self.map_np.shape
        edge_usage_matrix = edge_usage_matrix.reshape(h, w, 4)
        wait_usage_matrix = wait_usage_matrix.reshape(h, w, 1)
        edge_weight_matrix = edge_weight_matrix.reshape(h, w, 4)
        wait_cost_matrix = wait_cost_matrix.reshape(h, w, 1)
        input = np.concatenate(
            [
                edge_usage_matrix,
                wait_usage_matrix,
                edge_weight_matrix,
                wait_cost_matrix,
            ],
            axis=2,
            dtype=np.float32,
        )
        input = np.moveaxis(input, 2, 0)
        return input

    def _run_sim(self, init_weight=False):
        """Run one simulation on the current edge weights and wait costs

        Args:
            init_weight (bool, optional): Whether the current simulation is on
                the initial weights. Defaults to False.

        """
        curr_map_json = copy.deepcopy(self.map_json)
        curr_map_json["weight"] = True
        curr_map_json["optimize_wait"] = self.config.optimize_wait

        # Initial weights are assumed to be valid and optimize_waits = True
        if init_weight:
            edge_weights = self.curr_edge_weights.tolist()
            wait_costs = self.curr_wait_costs.tolist()
            curr_map_json["weights"] = [*wait_costs, *edge_weights]
            # if self.config.optimize_wait:
            # else:
            #     all_weights = [self.curr_wait_costs, *edge_weights]
            # curr_map_json["weights"] = all_weights
        else:
            edge_weights = min_max_normalize(self.curr_edge_weights, self.lb,
                                             self.ub).tolist()
            if self.config.optimize_wait:
                wait_costs = min_max_normalize(self.curr_wait_costs, self.lb,
                                               self.ub).tolist()
                curr_map_json["weights"] = [*wait_costs, *edge_weights]
            else:
                all_weights = [self.curr_wait_costs, *edge_weights]
                all_weights = min_max_normalize(all_weights, self.lb, self.ub)
                curr_map_json["weights"] = all_weights.tolist()

        sim_seed = self.rng.integers(10000)
        output = os.path.join(self.eval_logdir,
                              f"piu-iter_{self.i}-seed={sim_seed}")
        kwargs = {
            "map": json.dumps(curr_map_json),
            "output": output,
            "scenario": self.config.scenario,
            "task": self.config.task,
            "agentNum": self.num_agents,
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

        result_jsonstr = warehouse_sim.run(**kwargs)
        result = json.loads(result_jsonstr)
        return result

    def step(self, action):
        self.i += 1  # increment timestep

        # The environment is fully observable, so the observation is the
        # current edge weights/wait costs

        if self.config.optimize_wait:
            wait_cost_update_vals = action[:self.n_valid_vertices]
            edge_weight_update_vals = action[self.n_valid_vertices:]
        else:
            wait_cost_update_vals = action[0]
            edge_weight_update_vals = action[1:]
        self.curr_wait_costs = wait_cost_update_vals
        self.curr_edge_weights = edge_weight_update_vals

        # Reward is difference between new throughput and current throughput
        result = self._run_sim()
        new_throughput = result["throughput"]
        reward = new_throughput - self.curr_throughput
        self.curr_throughput = new_throughput

        # Stop early if the decrease in throughput is too much from initial
        # result
        # if new_throughput - self.init_throughput < 0 and np.abs(
        #         reward) / self.init_throughput >= 0.1:
        if new_throughput - self.init_throughput < 0:
            terminated = True
            print(f"Stop early at iter {self.i}!")
        # Otherwise terminated/truncate only if max iter is passed
        else:
            terminated = self.i >= self.max_iter
        truncated = terminated

        # Info includes the results
        info = {
            "result": result,
            "curr_wait_costs": self.curr_wait_costs,
            "curr_edge_weights": self.curr_edge_weights,
        }

        return self._gen_obs(result), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        init_result = info["result"]
        return self._gen_obs(init_result), info
