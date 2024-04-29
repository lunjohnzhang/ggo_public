import time
import random
import logging
import pathlib
import json

import traceback
import numpy as np

from typing import List, Callable
from env_search.competition.competition_result import CompetitionResult
from env_search.utils.worker_state import get_competition_module
from env_search.competition.module import CompetitionModule

logger = logging.getLogger(__name__)


def run_competition(
    edge_weights_json: str,
    wait_costs_json: str,
    eval_logdir: pathlib.Path,
    sim_seed: int,
    map_id: int,
    eval_id: int,
) -> CompetitionResult:
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
    """Grabs the competition module and evaluates level n_evals times."""

    if edge_weights_json is None:
        logger.info("Evaluating failed layout. Skipping")
        result = {"success": False}
        return result

    start = time.time()

    logger.info("seeding global randomness")
    np.random.seed(sim_seed // np.int32(4))
    random.seed(sim_seed // np.int32(2))

    logger.info("run competition with seed %d", sim_seed)
    competition_module = get_competition_module()

    try:
        result = competition_module.evaluate(
            edge_weights_json=edge_weights_json,
            wait_costs_json=wait_costs_json,
            eval_logdir=eval_logdir,
            sim_seed=sim_seed,
            map_id=map_id,
            eval_id=eval_id,
        )
        result["success"] = True
    except TimeoutError as e:
        edge_weights_json = json.loads(edge_weights_json)
        if wait_costs_json is not None:
            wait_costs_json = json.loads(wait_costs_json)
        logger.warning(f"evaluate failed")
        logger.info(f"The edge weights were {edge_weights_json}")
        logger.info(f"The wait costs were {wait_costs_json}")
        result = {"success": False}

    logger.info("run_competition done after %f sec", time.time() - start)

    return result


def run_competition_iterative_update(
    n_valid_edges: int,
    n_valid_vertices: int,
    eval_logdir: str,
    model_params: np.ndarray,
    seed: int,
):
    start = time.time()

    competition_module = get_competition_module()

    result, all_weights = competition_module.evaluate_iterative_update(
        model_params,
        eval_logdir,
        n_valid_edges,
        n_valid_vertices,
        seed,
    )

    logger.info("run_competition_iterative_update done after %f sec",
                time.time() - start)

    return result, all_weights


def process_competition_eval_result(
    edge_weights,
    wait_costs,
    curr_result_json: List[dict],
    n_evals: int,
    map_id: int,
):
    start = time.time()

    competition_module = get_competition_module()

    results = competition_module.process_eval_result(
        edge_weights=edge_weights,
        wait_costs=wait_costs,
        curr_result_json=curr_result_json,
        n_evals=n_evals,
        map_id=map_id,
    )
    logger.info("process_competition_eval_result done after %f sec",
                time.time() - start)

    return results
