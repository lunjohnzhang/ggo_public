import os
import fire
import json
import numpy as np
import matplotlib.pyplot as plt

from env_search.utils import MAP_DIR, read_in_kiva_map


def plot_dist(weights, ax):
    # Create a histogram using the axes object
    ax.hist(weights, bins=100, edgecolor='black', alpha=0.7)

    # Title and labels using the axes object
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_xlabel("Wait costs", fontsize=25)
    ax.set_ylabel("Frequency", fontsize=25)


def plot_edge_weight_dist(
    map_filepath,
    store_dir=MAP_DIR,
    lb=0.1,
    ub=100,
):
    with open(map_filepath, "r") as f:
        weights = json.load(f)["weights"][:819]
    _, map_name = read_in_kiva_map(map_filepath)

    # Create a new figure and axes
    fig, ax = plt.subplots()
    fig_norm, ax_norm = plt.subplots()
    plot_dist(weights, ax)
    fig.tight_layout()

    # Save the figure to disk
    fig.savefig(os.path.join(store_dir, f"{map_name}_wait_costs.png"))

    # Plot normalized weights
    weights = np.array(weights)
    min_sols = np.min(weights)
    max_sols = np.max(weights)
    norm_weights = lb + (weights - min_sols) * (ub - lb) / (max_sols -
                                                            min_sols)
    plot_dist(norm_weights, ax_norm)

    fig_norm.tight_layout()

    # Save the figure to disk
    fig_norm.savefig(os.path.join(store_dir, f"{map_name}_edge_dist_norm.png"))


if __name__ == "__main__":
    fire.Fire(plot_edge_weight_dist)
