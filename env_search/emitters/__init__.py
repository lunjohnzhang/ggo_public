"""pyribs-compliant emitters."""
import gin
import ribs

from env_search.emitters.map_elites_baseline_emitter import MapElitesBaselineWarehouseEmitter
from env_search.emitters.random_emitter import RandomEmitter
from env_search.emitters.evolution_strategy_emitter import EvolutionStrategyEmitter
from env_search.emitters.opt._cma_es import CMAEvolutionStrategy
from env_search.emitters.pycma_es_emitter import PyCMAEmitter

__all__ = [
    "GaussianEmitter",
    "EvolutionStrategyEmitter",
    "MapElitesBaselineWarehouseEmitter",
    "RandomEmitter",
    "CMAEvolutionStrategy",
    "PyCMAEmitter",
]


@gin.configurable
class GaussianEmitter(ribs.emitters.GaussianEmitter):
    """gin-configurable version of pyribs GaussianEmitter."""

    def ask(self):
        # Return addition None to cope with parent sol API of
        # MapElitesBaselineWarehouseEmitter
        return super().ask(), None


@gin.configurable
class IsoLineEmitter(ribs.emitters.IsoLineEmitter):
    """gin-configurable version of pyribs IsoLineEmitter."""

    def ask(self):
        # Return addition None to cope with parent sol API of
        # MapElitesBaselineWarehouseEmitter
        return super().ask(), None
