"""Functions for managing worker state.

In general, one uses these by first calling init_* or set_* to create the
attribute, then calling get_* to retrieve the corresponding value.
"""
from functools import partial

from dask.distributed import get_worker

from env_search.warehouse.config import WarehouseConfig
from env_search.warehouse.module import WarehouseModule
from env_search.competition.config import CompetitionConfig
from env_search.competition.module import CompetitionModule

#
# Generic
#


def set_worker_state(key: str, val: object):
    """Sets worker_state[key] = val"""
    worker = get_worker()
    setattr(worker, key, val)


def get_worker_state(key: str) -> object:
    """Retrieves worker_state[key]"""
    worker = get_worker()
    return getattr(worker, key)


#
# Warehouse module
#

WAREHOUSE_MOD_ATTR = "warehouse_module"


def init_warehouse_module(config: WarehouseConfig):
    """Initializes this worker's warehouse module."""
    set_worker_state(WAREHOUSE_MOD_ATTR, WarehouseModule(config))


def get_warehouse_module() -> WarehouseModule:
    """Retrieves this worker's warehouse module."""
    return get_worker_state(WAREHOUSE_MOD_ATTR)


#
# Competition module
#

COMPETITION_MOD_ATTR = "competition_module"


def init_competition_module(config: CompetitionConfig):
    """Initializes this worker's competition module."""
    set_worker_state(COMPETITION_MOD_ATTR, CompetitionModule(config))


def get_competition_module() -> CompetitionModule:
    """Retrieves this worker's competition module."""
    return get_worker_state(COMPETITION_MOD_ATTR)
