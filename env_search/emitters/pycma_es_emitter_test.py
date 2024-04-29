import numpy as np
import cma

from pprint import pprint
from env_search.archives.grid_archive import GridArchive
from env_search.emitters.pycma_es_emitter import PyCMAEmitter

if __name__ == "__main__":
    sol_dim = 1000
    archive = GridArchive(
        solution_dim=sol_dim,
        dims=[25, 25],
        ranges=[[0, 10], [0, 10]],
    )
    pprint(cma.CMAOptions())
    emitter = PyCMAEmitter(
        archive,
        x0=np.ones(sol_dim),
        sigma0=5,
        bounds=[[0.1, 100]] * sol_dim,
        batch_size=10,
    )
    sols, _ = emitter.ask()
    emitter.tell(
        sols,
        np.random.rand(10),
        np.random.rand(10, 2),
        np.ones(10),
        np.ones(10),
    )
