import gin
import ribs
import numpy as np

from ribs._utils import readonly
from threadpoolctl import threadpool_limits


@gin.configurable
class CMAEvolutionStrategy(ribs.emitters.opt.CMAEvolutionStrategy):
    """gin-configurable version of pyribs CMAEvolutionStrategy. Also implements
    some Lamarckian bound handling approaches, described in this paper:
    https://www.sciencedirect.com/science/article/pii/S2210650219301622

    Args:
        sigma0 (float): Initial step size.
        batch_size (int): Number of solutions to evaluate at a time. If None, we
            calculate a default batch size based on solution_dim.
        solution_dim (int): Size of the solution space.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
        bound_handle (str):
            None: no bound handle
            projection: project solutions to bounds
    """

    def __init__(  # pylint: disable = super-init-not-called
            self,
            sigma0,
            solution_dim,
            batch_size=None,
            seed=None,
            dtype=np.float64,
            bound_handle=None):
        super().__init__(
            sigma0=sigma0,
            solution_dim=solution_dim,
            batch_size=batch_size,
            seed=seed,
            dtype=dtype,
        )
        self._bound_handle = bound_handle
        self._n_restart_from_area = 0

    def check_stop(self, ranking_values):
        """Checks if the optimization should stop and be reset.

        Tolerances come from CMA-ES.

        Args:
            ranking_values (np.ndarray): Array of objective values of the
                solutions, sorted in the same order that the solutions were
                sorted when passed to ``tell()``.

        Returns:
            True if any of the stopping conditions are satisfied.
        """
        if self.cov.condition_number > 1e14:
            return True

        # Area of distribution too small.
        area = self.sigma * np.sqrt(np.max(self.cov.eigenvalues))
        if area < 1e-11:
            self._n_restart_from_area += 1
            return True

        # Fitness is too flat (only applies if there are at least 2 parents).
        # NOTE: We use norm here because we may have multiple ranking values.
        if (len(ranking_values) >= 2 and
                np.linalg.norm(ranking_values[0] - ranking_values[-1]) < 1e-12):
            return True

        return False

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def ask(self, lower_bounds, upper_bounds, batch_size=None):
        """Samples new solutions from the Gaussian distribution.

        Args:
            lower_bounds (float or np.ndarray): scalar or (solution_dim,) array
                indicating lower bounds of the solution space. Scalars specify
                the same bound for the entire space, while arrays specify a
                bound for each dimension. Pass -np.inf in the array or scalar to
                indicated unbounded space.
            upper_bounds (float or np.ndarray): Same as above, but for upper
                bounds (and pass np.inf instead of -np.inf).
            batch_size (int): batch size of the sample. Defaults to
                ``self.batch_size``.
        """
        if batch_size is None:
            batch_size = self.batch_size

        self._solutions = np.empty((batch_size, self.solution_dim),
                                   dtype=self.dtype)
        self.cov.update_eigensystem(self.current_eval, self.lazy_gap_evals)
        transform_mat = self.cov.eigenbasis * np.sqrt(self.cov.eigenvalues)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds or until `max_iter` is reached. After
        # that, clip the remaining solutions within the bounds
        remaining_indices = np.arange(batch_size)
        iter = 0
        max_iter = 100
        while len(remaining_indices) > 0 and iter < max_iter:
            unscaled_params = self._rng.normal(
                0.0,
                self.sigma,
                (len(remaining_indices), self.solution_dim),
            ).astype(self.dtype)
            new_solutions, out_of_bounds = self._transform_and_check_sol(
                unscaled_params, transform_mat, self.mean, lower_bounds,
                upper_bounds)
            self._solutions[remaining_indices] = new_solutions

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each value in each solution is
            # out of bounds).
            remaining_indices = remaining_indices[np.any(out_of_bounds, axis=1)]
            iter += 1

        # print(
        #     f"Number of invalid solutions after resampling: {len(remaining_indices)}"
        # )
        # Handle bounds, all bound handles here would be `Lamarckian` approach
        # where `repaired` solutions are passes to tell method.
        if self._bound_handle is not None:
            if self._bound_handle == "projection":
                self._solutions = np.clip(self._solutions, lower_bounds,
                                          upper_bounds)

        return readonly(self._solutions)


@gin.configurable
class LMMAEvolutionStrategy(ribs.emitters.opt.LMMAEvolutionStrategy):
    """LM-MA-ES optimizer for use with emitters.

    Refer to :class:`EvolutionStrategyBase` for usage instruction.

    Args:
        sigma0 (float): Initial step size.
        batch_size (int): Number of solutions to evaluate at a time. If None, we
            calculate a default batch size based on solution_dim.
        solution_dim (int): Size of the solution space.
        seed (int): Seed for the random number generator.
        dtype (str or data-type): Data type of solutions.
        n_vectors (int): Number of vectors to use in the approximation. If None,
            this defaults to be equal to the batch size.
    """

    def __init__(  # pylint: disable = super-init-not-called
            self,
            sigma0,
            solution_dim,
            batch_size=None,
            seed=None,
            dtype=np.float64,
            n_vectors=None):
        super().__init__(
            sigma0=sigma0,
            solution_dim=solution_dim,
            batch_size=batch_size,
            seed=seed,
            dtype=dtype,
            n_vectors=n_vectors,
        )
        self._n_restart_from_area = 0

    # Limit OpenBLAS to single thread. This is typically faster than
    # multithreading because our data is too small.
    @threadpool_limits.wrap(limits=1, user_api="blas")
    def ask(self, lower_bounds, upper_bounds, batch_size=None):
        """Samples new solutions from the Gaussian distribution.

        Args:
            lower_bounds (float or np.ndarray): scalar or (solution_dim,) array
                indicating lower bounds of the solution space. Scalars specify
                the same bound for the entire space, while arrays specify a
                bound for each dimension. Pass -np.inf in the array or scalar to
                indicated unbounded space.
            upper_bounds (float or np.ndarray): Same as above, but for upper
                bounds (and pass np.inf instead of -np.inf).
            batch_size (int): batch size of the sample. Defaults to
                ``self.batch_size``.
        """
        # NOTE: The LM-MA-ES uses mirror sampling by default, but we do not.
        if batch_size is None:
            batch_size = self.batch_size

        self._solutions = np.empty((batch_size, self.solution_dim),
                                   dtype=self.dtype)
        self._solution_z = np.empty((batch_size, self.solution_dim),
                                    dtype=self.dtype)

        # Resampling method for bound constraints -> sample new solutions until
        # all solutions are within bounds.
        remaining_indices = np.arange(batch_size)
        iter = 0
        max_iter = 100
        while len(remaining_indices) > 0 and iter < max_iter:
            z = self._rng.standard_normal(
                (len(remaining_indices), self.solution_dim))  # (_, n)
            self._solution_z[remaining_indices] = z

            new_solutions, out_of_bounds = self._transform_and_check_sol(
                z, min(self.current_gens, self.n_vectors), self.cd, self.m,
                self.mean, self.sigma, lower_bounds, upper_bounds)
            self._solutions[remaining_indices] = new_solutions

            # Find indices in remaining_indices that are still out of bounds
            # (out_of_bounds indicates whether each value in each solution is
            # out of bounds).
            remaining_indices = remaining_indices[np.any(out_of_bounds, axis=1)]
            iter += 1

        return readonly(self._solutions)
