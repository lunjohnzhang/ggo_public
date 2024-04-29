import copy
from typing import Optional

import gin
import numpy as np
import ribs
from ribs.emitters import EmitterBase


@gin.configurable(denylist=["archive", "x0", "seed"])
class MapElitesBaselineWarehouseEmitter(EmitterBase):
    """Implementation of MAP-Elites which generates solutions corresponding to
    map layout.

    Args:
        archive: Archive to store the solutions.
        x0: Initial solution. Only used for solution_dim.
        bounds: Bounds of the solution space. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to
            specify the bounds for each dim. Each element in this array-like can
            be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound. (default: None)
        seed: Random seed. (default None)
        num_objects: Solutions will be generated as ints between
            [1, num_objects]
        batch_size: Number of solutions to return in :meth:`ask`.
        initial_population: Size of the initial population before starting to
            mutate elites from the archive.
        mutation_k: Number of positions in the solution to mutate. Should be
            less than solution_dim.
        geometric_k: Whether to vary k geometrically. If it is True,
            `mutation_k` will be ignored.
        max_n_shelf: max number of shelves(index 1).
        min_n_shelf: min number of shelves(index 1).
    """

    def __init__(
        self,
        archive: ribs.archives.ArchiveBase,
        x0: np.ndarray,
        bounds: Optional["array-like"] = None,  # type: ignore
        seed: int = None,
        num_objects: int = gin.REQUIRED,
        batch_size: int = gin.REQUIRED,
        initial_population: int = gin.REQUIRED,
        mutation_k: int = gin.REQUIRED,
        geometric_k: bool = gin.REQUIRED,
        max_n_shelf: float = gin.REQUIRED,
        min_n_shelf: float = gin.REQUIRED,
    ):
        solution_dim = len(x0)
        super().__init__(
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.num_objects = num_objects
        self.initial_population = initial_population
        self.mutation_k = mutation_k
        self.geometric_k = geometric_k
        self.max_n_shelf = max_n_shelf
        self.min_n_shelf = min_n_shelf

        if not self.geometric_k:
            assert solution_dim >= self.mutation_k

        # # When we know the exact number of shelves and we only have shelf or
        # # floor, k will be used to switch randomly pairs of 0's and 1's
        # if self.max_n_shelf == self.min_n_shelf and self.num_objects == 2:
        #     if not self.geometric_k:
        #         assert self.min_n_shelf >= self.mutation_k
        #         assert self.solution_dim - self.min_n_shelf >= self.mutation_k

        self.sols_emitted = 0

    def ask(self):
        if self.sols_emitted < self.initial_population:
            sols = self.rng.integers(low=1,
                                     high=self.num_objects,
                                     size=(self.batch_size, self.solution_dim),
                                     endpoint=True)

            self.sols_emitted += self.batch_size
            return np.array(sols), None

        # Mutate current solutions
        else:
            sols = []

            # select k spots randomly without replacement
            # and calculate the random replacement values
            curr_k = self.sample_k(self.solution_dim)
            idx_array = np.tile(np.arange(self.solution_dim),
                                (self.batch_size, 1))
            mutate_idxs = self.rng.permuted(idx_array, axis=1)[:, :curr_k]
            mutate_vals = self.rng.integers(low=1,
                                            high=self.num_objects,
                                            size=(self.batch_size, curr_k),
                                            endpoint=True)

            parent_sols = self.archive.sample_elites(self.batch_size)
            for i in range(self.batch_size):
                parent_sol = parent_sols.solution_batch[i]
                sol = copy.deepcopy(parent_sol.astype(int))
                # Replace with random values
                sol[mutate_idxs[i]] = mutate_vals[i]
                sols.append(sol)

            self.sols_emitted += self.batch_size
            return np.array(sols), parent_sols.solution_batch

    def sample_k(self, max_k):
        if self.geometric_k:
            curr_k = self.rng.geometric(p=0.5)
            # Clip k if necessary
            if curr_k > max_k:
                curr_k = max_k
        else:
            curr_k = self.mutation_k
        return curr_k
