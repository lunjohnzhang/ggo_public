"""Visualizes throughput vs. n_agents from multiple experiments.

Usage:
    python env_search/analysis/throughput_vs_n_agents_cross.py --all_logdirs_plot <all_logdirs_plot>
"""
import os
import fire
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from env_search.analysis.throughput_vs_n_agents import throughput_vs_n_agents

mpl.use("agg")

# set matplotlib params
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def throughput_vs_n_agents_cross(
    all_logdirs_plot,
    ax=None,
    add_legend=True,
    save_fig=True,
):
    min_agent_num = np.inf
    max_agent_num = -1
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    for i, logdirs_plot_dir in enumerate(sorted(os.listdir(all_logdirs_plot))):
        logdirs_plot = os.path.join(all_logdirs_plot, logdirs_plot_dir)
        if not os.path.isdir(logdirs_plot):
            continue
        plot_result = throughput_vs_n_agents(logdirs_plot, ax=ax)

        # Skip if there is nothing in the current logdir
        if plot_result is None:
            continue

        # Otherwise, process the plotted result
        agent_nums, y_min, y_max, meta = plot_result
        if agent_nums[-1] > max_agent_num:
            max_agent_num = agent_nums[-1]
        if agent_nums[0] < min_agent_num:
            min_agent_num = agent_nums[0]
    # ax.set_xlim(agent_nums[0], agent_nums[-1])
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Throughput", fontsize=45)
    ax.set_xlabel("Number of Agents", fontsize=45)
    # ax.grid()
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.tick_params(axis='both', which='minor', labelsize=15)

    # Add a vertical line at the number of agents used to optimize the layout
    map_size = meta["map_size"]
    mode = meta["mode"]

    n_agent_vertical = None
    if map_size == "small":
        if mode == "w":
            n_agent_vertical = [60]
        elif mode == "r":
            n_agent_vertical = [88]
    elif map_size == "medium":
        n_agent_vertical = [90]
    elif map_size == "large":
        n_agent_vertical = [220, 400]
        # n_agent_vertical = [150]
        # n_agent_vertical = [220]
    elif map_size == "32x32":
        n_agent_vertical = [400]
    elif map_size == "64x64":
        n_agent_vertical = [1500]
    elif map_size == "empty-48-48":
        n_agent_vertical = [1000]
    elif map_size == "maze-32-32-4":
        n_agent_vertical = [400]
    elif map_size == "random-64-64-20":
        n_agent_vertical = [1500]
    elif map_size == "den312d":
        n_agent_vertical = [1200]

    # Draw vertical line
    if n_agent_vertical is not None:
        for v in n_agent_vertical:
            ax.axvline(x=v, color="black", linewidth=2)

    range_x = [min_agent_num, max_agent_num]
    range_y = [y_min, y_max]
    if map_size in ["small", "medium"]:
        ax.set_xticks([
            range_x[0],
            np.mean([range_x]),
            *n_agent_vertical,
            range_x[1],
        ])
        ax.set_xticklabels(
            [
                range_x[0],
                np.mean([range_x], dtype=int),
                *n_agent_vertical,
                range_x[1],
            ],
            rotation=0,
            fontsize=35,
        )
    # elif map_size == "large":
    #     ax.set_xticks([
    #         range_x[0],
    #         # 100,
    #         *n_agent_vertical,
    #         range_x[1],
    #     ])
    #     ax.set_xticklabels(
    #         [
    #             range_x[0],
    #             # 100,
    #             *n_agent_vertical,
    #             range_x[1],
    #         ],
    #         rotation=0,
    #         fontsize=35,
    #     )
    elif map_size in [
            "large",
            "32x32",
            "64x64",
            "empty-48-48",
            "maze-32-32-4",
            "random-64-64-20",
            "den312d",
    ]:
        ax.set_xticks([
            range_x[0],
            # 100,
            *n_agent_vertical,
            range_x[1],
        ])
        ax.set_xticklabels(
            [
                range_x[0],
                # 100,
                *n_agent_vertical,
                range_x[1],
            ],
            rotation=0,
            fontsize=35,
        )
    else:
        range_x_mid = np.mean(range_x)
        if range_x_mid.is_integer():
            range_x_mid = int(range_x_mid)
        ax.set_xticks([range_x[0], range_x_mid, range_x[1]])
        ax.set_xticklabels(
            [range_x[0], range_x_mid, range_x[1]],
            fontsize=35,
        )
    range_y_mid = np.mean(range_y)
    if range_y_mid.is_integer():
        range_y_mid = int(range_y_mid)
    ax.set_yticks([range_y[0], range_y_mid, range_y[1]])
    ax.set_yticklabels(
        [range_y[0], range_y_mid, range_y[1]],
        fontsize=35,
    )

    # Legend
    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles,
            labels,
            loc="lower left",
            ncol=1,
            fontsize=25,
            mode="expand",
            bbox_to_anchor=(0, 1.02, 1, 0.2),  # for ncols=2
            # borderaxespad=0,)
        )

        # # Put legend inside
        # legend = ax.legend(
        #     handles,
        #     labels,
        #     loc="lower left",
        #     ncol=2,
        #     fontsize=25,
        #     mode="expand",
        #     bbox_to_anchor=(0, 0.75, 1, 0.2),
        # )

        # For front fig
        # order = [1, 0]
        # ax.legend(
        #     [handles[idx] for idx in order],
        #     [labels[idx] for idx in order],
        #     # loc="lower left",
        #     ncol=1,
        #     fontsize=35,
        #     frameon=False,
        #     # mode="expand",
        #     # bbox_to_anchor=(0, 1.02, 1, 0.2),  # for ncols=2
        #     # borderaxespad=0,)
        # )

        # For r-mode less agents
        # legend = ax.legend(
        #     handles,
        #     labels,
        #     # loc="lower left",
        #     ncol=1,
        #     fontsize=35,
        #     borderaxespad=0,
        #     bbox_to_anchor=(1.04, 1),  # for ncols=2
        # )

        for line in legend.get_lines():
            line.set_linewidth(2.0)

    if save_fig:
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                all_logdirs_plot,
                f"agent_num_{mode}_{map_size}.pdf",
            ),
            dpi=300,
            bbox_inches='tight',
        )

        fig.savefig(
            os.path.join(
                all_logdirs_plot,
                f"agent_num_{mode}_{map_size}.png",
            ),
            dpi=300,
            bbox_inches='tight',
        )


if __name__ == "__main__":
    fire.Fire(throughput_vs_n_agents_cross)
