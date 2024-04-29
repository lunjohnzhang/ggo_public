"""Visualizes throughput vs. n_agents from an experiment.

Usage:
    python env_search/analysis/throughput_vs_n_agents.py --logdirs_plot <log_dir_plot>
"""
import os
import json
from typing import List

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
import scipy.stats as st
import pandas as pd
# import warnings
# warnings.filterwarnings("error")

from env_search.analysis.utils import (algo_name_map, get_color, get_line_style,
                                       get_marker)

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# color_map = {
#     "DSAGE": "green",
#     "MAP-Elites": "red",
#     # "ours": "red",
#     "Human": "blue",
#     "CMA-ME + NCA": "cyan",
#     "CMA-MAE + NCA": "gold",
# }


def add_item_to_dict_by_agent_num(to_dict, agent_num, element):
    if agent_num in to_dict:
        to_dict[agent_num].append(element)
    else:
        to_dict[agent_num] = [element]


def sort_and_get_vals_from_dict(the_dict):
    the_dict = sorted(the_dict.items())
    agent_nums = [agent_num for agent_num, _ in the_dict]
    all_vals = [vals for _, vals in the_dict]
    return agent_nums, all_vals


def compute_numerical(vals, all_success_vals):
    # Take the average, confidence interval and standard error
    all_vals = np.array(vals)
    all_success_vals = np.array(all_success_vals)
    assert all_vals.shape == all_success_vals.shape
    # breakpoint()
    mean_vals = np.mean(all_vals, axis=1)
    mean_vals_success = []
    sem_vals_success = []
    for i, curr_vals in enumerate(all_vals):
        # curr_vals = [x for x in curr_vals if x != 0]
        filtered_curr_vals = []
        for j, x in enumerate(curr_vals):
            if all_success_vals[i, j] == 1:
                filtered_curr_vals.append(x)

        if len(filtered_curr_vals) > 0:
            mean_vals_success.append(np.mean(filtered_curr_vals))

            if len(filtered_curr_vals) == 1:
                sem_vals_success.append(0)
            else:
                sem_vals_success.append(st.sem(filtered_curr_vals))
        else:
            mean_vals_success.append(np.nan)
            sem_vals_success.append(np.nan)

    cf_vals = st.t.interval(confidence=0.95,
                            df=all_vals.shape[1] - 1,
                            loc=mean_vals,
                            scale=st.sem(all_vals, axis=1) + 1e-8)
    sem_vals = st.sem(all_vals, axis=1)
    return mean_vals, cf_vals, sem_vals, mean_vals_success, sem_vals_success


def throughput_vs_n_agents(logdirs_plot: str, ax=None):
    with open(os.path.join(logdirs_plot, "meta.yaml"), "r") as f:
        meta = yaml.safe_load(f)

    algo_name = meta["algorithm"]
    map_size = meta["map_size"]
    mode = meta["mode"]
    map_from = meta["map_from"]
    n_agents_opt = meta.get("n_agents_opt", None)
    all_throughputs_dict = {}  # Raw throughput
    all_runtime_dict = {}
    all_success_dict = {}
    markevery = 1
    if map_size in ["45x47", "57x58", "69x69", "xxlarge"]:
        y_min = 0
        y_max = 10
        markevery = 2
    elif map_size in ["large"]:
        y_min = 0
        y_max = 8
        markevery = 2
    elif map_size in ["81x80", "93x91"]:
        y_min = 0
        y_max = 12
    elif map_size in ["32x32"]:
        y_min = 0
        y_max = 9
    elif map_size in ["64x64"]:
        y_min = 0
        y_max = 3.5
    elif map_size in ["small", "medium"]:
        y_min = 0
        y_max = 6
        markevery = 4
    elif map_size in ["maze-32-32-4"]:
        y_min = 0
        y_max = 1.5
    elif map_size in ["empty-48-48"]:
        y_min = 0
        y_max = 32
    elif map_size in ["den312d"]:
        y_min = 0
        y_max = 5
    else:
        y_min = 0
        y_max = 10
    for logdir_f in os.listdir(logdirs_plot):
        logdir = os.path.join(logdirs_plot, logdir_f)
        if not os.path.isdir(logdir):
            continue
        results_dir = os.path.join(logdir, "results")
        # agent_nums = []
        # throughputs = []
        for sim_dir in os.listdir(results_dir):
            sim_dir_comp = os.path.join(results_dir, sim_dir)
            config_file = os.path.join(sim_dir_comp, "config.json")
            result_file = os.path.join(sim_dir_comp, "result.json")

            if os.path.exists(config_file) and os.path.exists(result_file):

                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)

                    with open(result_file, "r") as f:
                        result = json.load(f)
                except json.decoder.JSONDecodeError:
                    print(result_file)

                congested = result[
                    "congested"] if "congested" in result else False
                agent_num = config[
                    "agentNum"] if "agentNum" in config else config["num_agents"]

                # Only consider the uncongested simulations
                throughput = result["throughput"]  # if not congested else 0
                runtime = result[
                    "cpu_runtime"] if "cpu_runtime" in result else 0  # if not congested else 0
                success = 1 if not congested else 0
                # agent_nums.append(agent_num)
                # throughputs.append(throughput)

                add_item_to_dict_by_agent_num(
                    all_throughputs_dict,
                    agent_num,
                    throughput,
                )
                add_item_to_dict_by_agent_num(
                    all_runtime_dict,
                    agent_num,
                    runtime,
                )
                add_item_to_dict_by_agent_num(
                    all_success_dict,
                    agent_num,
                    success,
                )

            else:
                print(f"Result of {sim_dir} is missing")

        # sort_idx = np.argsort(agent_nums)
        # agent_nums = np.array(agent_nums)[sort_idx]
        # throughputs = np.array(throughputs)[sort_idx]

        # all_throughputs_dict.append(throughputs)

    # Nothing in the logdir, skip
    if not all_throughputs_dict:
        return None

    # all_throughputs_dict = sorted(all_throughputs_dict.items())
    # agent_nums = [agent_num for agent_num, _ in all_throughputs_dict]
    # all_throughputs_vals = [
    #     throughputs for _, throughputs in all_throughputs_dict
    # ]

    agent_nums, all_throughputs_vals = sort_and_get_vals_from_dict(
        all_throughputs_dict)
    _, all_runtime_vals = sort_and_get_vals_from_dict(all_runtime_dict)
    _, all_success_vals = sort_and_get_vals_from_dict(all_success_dict)

    save_fig = False
    if ax is None:
        save_fig = True
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Take the average, confidence interval and standard error
    # all_throughputs = np.array(all_throughputs_vals)
    # mean_throughputs = np.mean(all_throughputs, axis=1)
    # cf_throughputs = st.t.interval(confidence=0.95,
    #                                df=len(all_throughputs) - 1,
    #                                loc=mean_throughputs,
    #                                scale=st.sem(all_throughputs, axis=1) + 1e-8)
    # sem_throughputs = st.sem(all_throughputs, axis=1)

    (
        mean_throughputs,
        cf_throughputs,
        sem_throughputs,
        mean_throughputs_success,
        sem_throughputs_success,
    ) = compute_numerical(all_throughputs_vals, all_success_vals)

    (
        mean_runtime,
        _,
        sem_runtime,
        mean_runtime_success,
        sem_runtime_success,
    ) = compute_numerical(all_runtime_vals, all_success_vals)

    all_success_vals = np.array(all_success_vals)
    success_rates = np.sum(all_success_vals, axis=1) / all_success_vals.shape[1]
    # breakpoint()

    color = get_color(map_from, algo_name, n_agents_opt)
    line_style = get_line_style(map_from, algo_name)
    marker = get_marker(map_from, algo_name)
    label = f"{algo_name_map[algo_name]} + {map_from}"
    # label = f"{map_from}"
    if n_agents_opt is not None:
        label += f"({n_agents_opt} agents)"

    if "N_e_piu" in label:
        plt.rc("text", usetex=True)
        markevery=1
        label = label.replace("N_e_piu", r"$N_{e\_piu}$")

    ax.plot(
        agent_nums,
        mean_throughputs,
        marker=marker,
        color=color,
        label=label,
        markersize=15,
        linestyle=line_style,
        markevery=markevery,
        # label=f"{map_from}",
    )
    ax.fill_between(
        agent_nums,
        cf_throughputs[1],
        cf_throughputs[0],
        alpha=0.3,
        color=color,
    )

    if save_fig:
        ax.set_ylabel("Throughput", fontsize=25)
        ax.set_xlabel("Number of Agents", fontsize=25)
        ax.set_ylim(y_min, y_max)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=15)

        ax.figure.tight_layout()
        fig.savefig(
            os.path.join(
                logdirs_plot,
                f"throughput_agentNum_{algo_name}_{map_size}_{mode}.png",
            ),
            dpi=300,
        )

    # Create numerical result
    numerical_result = {}
    numerical_result["agent_num"] = agent_nums
    numerical_result["mean_throughput"] = mean_throughputs
    numerical_result["mean_throughput_success"] = mean_throughputs_success
    numerical_result["sem_throughput"] = sem_throughputs
    numerical_result["sem_throughputs_success"] = sem_throughputs_success
    numerical_result["mean_runtime"] = mean_runtime
    numerical_result["mean_runtime_success"] = mean_runtime_success
    numerical_result["sem_runtime"] = sem_runtime
    numerical_result["sem_runtime_success"] = sem_runtime_success
    numerical_result["success_rate"] = success_rates
    numerical_result_df = pd.DataFrame(numerical_result)
    numerical_result_df.to_csv(
        os.path.join(
            logdirs_plot,
            f"numerical_{algo_name}_{map_size}_{mode}.csv",
        ))

    return agent_nums, y_min, y_max, meta


if __name__ == "__main__":
    fire.Fire(throughput_vs_n_agents)
