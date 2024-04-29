import gin

from dataclasses import dataclass
from typing import Collection, Optional, Tuple, List, Callable, Dict


@gin.configurable
@dataclass
class CompetitionConfig:
    """
    Config competition simulation

    Args:
        measure_names (list[str]): list of names of measures
        aggregation_type (str): aggregation over `n_evals` results

        # Simulation
        obj_type (str): type of objective one of ["throughput"]
        simulation_time (int): number of timestep to run simulation
        map_path (str): filepath to the map to be optimized
        num_agents (int): number of agents

        # PIBT (or rather WPPL without LNS) simulation
        plan_time_limit (int): time limit of planning at each timestep
        preprocess_time_limit (int): time limit of preprocessing
        file_storage_path (str): path to save log files
        task_assignment_strategy (str): strategy of task assignment
        num_tasks_reveal (int): number of tasks revealed to the system in
                                advance
        gen_random (bool): if True, generate random tasks
        num_tasks (int): total number of tasks

        # PIU related
        bounds (Tuple): bound of the edge weights/wait costs
        iter_update_model_type (Callable): type of update model
        iter_update_max_iter (int): number of iterations to run PIU
        iter_update_n_sim (int): number of simulations to run in each iteration
                                 of PIU
        iter_update_mdl_kwargs (dict): kwargs of the update model

    """
    # Measures.
    measure_names: Collection[str] = gin.REQUIRED

    # Results.
    aggregation_type: str = gin.REQUIRED

    # Simulation
    obj_type: str = gin.REQUIRED
    simulation_time: int = gin.REQUIRED
    map_path: str = gin.REQUIRED
    num_agents: int = gin.REQUIRED
    plan_time_limit: int = 1
    preprocess_time_limit: int = 1800
    file_storage_path: str = "large_files"
    task_assignment_strategy: str = "roundrobin"
    num_tasks_reveal: int = 1
    gen_random: bool = True
    num_tasks: int = 100000

    # Iterative update. All params are optional b.c. iterative update is turned
    # off by default
    bounds: Tuple = None
    iter_update_model_type: Callable = None
    iter_update_max_iter: int = None
    iter_update_n_sim: int = 1
    iter_update_mdl_kwargs: Dict = None