import gin

from dataclasses import dataclass
from typing import Collection, Tuple, Callable


@gin.configurable
@dataclass
class WarehouseConfig:
    """
    Config warehouse simulation

    Args:
        measure_names (list[str]): list of names of measures
        aggregation_type (str): aggregation over `n_evals` results
        scenario (str): scenario (SORTING, KIVA, ONLINE, BEE)
        task (str): input task file

        cutoffTime (int): cutoff time (seconds)
        screen (int): screen option (0: none; 1: results; 2:all)
        solver (str): solver (LRA, PBS, WHCA, ECBS)
        id (bool): independence detection
        single_agent_solver (str): single-agent solver (ASTAR, SIPP)
        lazyP (bool): use lazy priority
        simulation_time (int): run simulation
        simulation_window (int): call the planner every simulation_window
                                 timesteps
        travel_time_window (int): consider the traffic jams within the
                                  given window
        planning_window (int): the planner outputs plans with first
                                     planning_window timesteps collision-free
        potential_function (str): potential function (NONE, SOC, IC)
        potential_threshold (int): potential threshold
        rotation (bool): consider rotation
        robust (int): k-robust (for now, only work for PBS)
        CAT (bool): use conflict-avoidance table
        hold_endpoints (bool): Hold endpoints from Ma et al, AAMAS 2017
        dummy_paths (bool): Find dummy paths from Liu et al, AAMAS 2019
        prioritize_start (bool): Prioritize waiting at start locations
        suboptimal_bound (int): Suboptimal bound for ECBS
        log (bool): save the search trees (and the priority trees)
        test (bool): whether under testing mode.
        use_warm_up (bool): if True, will use the warm-up procedure. In
                            particular, for the initial population, the solution
                            returned from hamming distance objective will be
                            used. For mutated solutions, the solution of the
                            parent will be used.
        save_result (bool): Whether to allow C++ save the result of simulation
        save_solver (bool): Whether to allow C++ save the result of solver
        save_heuristics_table (bool): Whether to allow C++ save the result of
                                      heuristics table
        stop_at_traffic_jam (bool): whether stop the simulation at traffic jam
        obj_type (str): type of objective
                        ("throughput",
                         "throughput_plus_n_shelf",
                         "throughput_minus_hamming_dist")
    """
    # Measures.
    measure_names: Collection[str] = gin.REQUIRED

    # Results.
    aggregation_type: str = gin.REQUIRED,

    # Simulation
    scenario: str = gin.REQUIRED,
    task: str = gin.REQUIRED,
    cutoffTime: int = gin.REQUIRED,
    screen: int = gin.REQUIRED,
    solver: str = gin.REQUIRED,
    id: bool = gin.REQUIRED,
    single_agent_solver: str = gin.REQUIRED,
    lazyP: bool = gin.REQUIRED,
    simulation_time: int = gin.REQUIRED,
    simulation_window: int = gin.REQUIRED,
    travel_time_window: int = gin.REQUIRED,
    planning_window: int = gin.REQUIRED,
    potential_function: str = gin.REQUIRED,
    potential_threshold: int = gin.REQUIRED,
    rotation: bool = gin.REQUIRED,
    robust: int = gin.REQUIRED,
    CAT: bool = gin.REQUIRED,
    hold_endpoints: bool = gin.REQUIRED,
    dummy_paths: bool = gin.REQUIRED,
    prioritize_start: bool = gin.REQUIRED,
    suboptimal_bound: int = gin.REQUIRED,
    log: bool = gin.REQUIRED,
    test: bool = gin.REQUIRED,
    use_warm_up: bool = gin.REQUIRED,
    hamming_only: bool = gin.REQUIRED,
    save_result: bool = gin.REQUIRED,
    save_solver: bool = gin.REQUIRED,
    save_heuristics_table: bool = gin.REQUIRED,
    stop_at_traffic_jam: bool = gin.REQUIRED,
    obj_type: str = gin.REQUIRED,
    left_w_weight: float = gin.REQUIRED,
    right_w_weight: float = gin.REQUIRED,
    hamming_obj_weight: float = 1
    repair_n_threads: int = 1
    optimize_wait: bool = True

    # Iterative update. All params are optional b.c. iterative update is turned
    # off by default
    bounds: Tuple = None
    iter_update_model_type: Callable = None
    iter_update_max_iter: int = 5
