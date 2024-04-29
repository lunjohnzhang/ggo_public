import os
import fire
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

from mpl_toolkits.axes_grid1 import AxesGrid
from env_search import MAP_DIR
from env_search.utils import (
    kiva_obj_types,
    kiva_env_str2number,
    kiva_env_number2str,
    read_in_kiva_map,
    kiva_color_map,
    kiva_get_highway_suggestion_graph,
    kiva_uncompress_edge_weights,
    kiva_uncompress_wait_costs,
    kiva_directions,
    competition_obj_types,
    get_n_valid_vertices,
    competition_env_str2number,
    read_in_competition_map,
)
from env_search.competition.update_model.utils import (
    Map, comp_uncompress_edge_matrix, comp_uncompress_vertex_matrix)

MAX_WEIGHT = 100


def convert_json_weights_to_matrix(map_filepath, domain, fill_value=0):
    """
    The map json has a `weight` entry that contains the wait costs/edge
    weights of the map as a list. This function converts the weights to their
    matrix format

    Args:
        map_filepath (str): json map path
        domain (str): domain of the map
    """
    with open(map_filepath, "r") as f:
        map_json = json.load(f)
    map_str_list = map_json["layout"]
    h, w = len(map_str_list), len(map_str_list[0])
    weights = map_json["weights"]
    optimize_wait = False
    if "optimize_wait" in map_json:
        optimize_wait = map_json["optimize_wait"]

    if domain == "kiva":
        map_np = kiva_env_str2number(map_str_list)
        n_valid_vertices = get_n_valid_vertices(map_np, domain=domain)
        block_idxs = [
            kiva_obj_types.index("@"),
        ]
        if not optimize_wait:
            wait_costs = np.zeros(n_valid_vertices)
            wait_costs[:] = weights[0]
            wait_costs_matrix = kiva_uncompress_wait_costs(
                map_np,
                wait_costs,
                block_idxs=block_idxs,
                fill_value=fill_value,
            )
            edge_weights_matrix = kiva_uncompress_edge_weights(
                map_np,
                weights[1:],
                block_idxs=block_idxs,
                fill_value=fill_value,
            )
        else:
            wait_costs = weights[:n_valid_vertices]
            edge_weights = weights[n_valid_vertices:]

            wait_costs_matrix = kiva_uncompress_wait_costs(
                map_np,
                wait_costs,
                block_idxs=block_idxs,
                fill_value=fill_value,
            )
            edge_weights_matrix = kiva_uncompress_edge_weights(
                map_np,
                edge_weights,
                block_idxs=block_idxs,
                fill_value=fill_value,
            )
    elif domain == "competition":
        comp_map = Map(map_filepath)
        map_np = competition_env_str2number(map_str_list)
        n_valid_vertices = get_n_valid_vertices(map_np, domain=domain)
        if optimize_wait:
            wait_costs = weights[:n_valid_vertices]
            edge_weights = weights[n_valid_vertices:]
            wait_costs_matrix = np.array(
                comp_uncompress_vertex_matrix(
                    comp_map,
                    wait_costs,
                    fill_value=fill_value,
                ))
            edge_weights_matrix = np.array(
                comp_uncompress_edge_matrix(
                    comp_map,
                    edge_weights,
                    fill_value=fill_value,
                ))
            edge_weights_matrix = edge_weights_matrix.reshape(h, w, 4)
            wait_costs_matrix = wait_costs_matrix.reshape(h, w)
        else:
            wait_costs = np.zeros(n_valid_vertices)
            wait_costs[:] = weights[0]
            wait_costs_matrix = comp_uncompress_vertex_matrix(
                comp_map, wait_costs, fill_value=fill_value)
            edge_weights_matrix = np.array(
                comp_uncompress_edge_matrix(
                    comp_map,
                    weights[1:],
                    fill_value=fill_value,
                ))
            edge_weights_matrix = edge_weights_matrix.reshape(h, w, 4)
    return wait_costs_matrix, edge_weights_matrix


def plot_highway_edge_weights(map_np,
                              weights,
                              map_name,
                              map_filepath=None,
                              store_dir=MAP_DIR,
                              domain="kiva",
                              optimize_wait=True):
    """Plot edge weights of highway as 4 heatmaps."""
    n_valid_vertices = get_n_valid_vertices(map_np, domain)
    if domain == "kiva":
        block_idxs = [
            kiva_obj_types.index("@"),
        ]
        optimized_wait_costs = weights[0]
        optimized_edge_weights = kiva_uncompress_edge_weights(
            map_np, weights[1:], block_idxs)
        optimized_wait_costs = [optimized_wait_costs] * n_valid_vertices
        optimized_wait_costs = kiva_uncompress_wait_costs(
            map_np, optimized_wait_costs, block_idxs)
    elif domain == "competition":
        comp_map = Map(map_filepath)
        block_idxs = [
            competition_obj_types.index("@"),
        ]
        if optimize_wait:
            optimized_wait_costs = comp_uncompress_vertex_matrix(
                comp_map, weights[:n_valid_vertices])
            optimized_wait_costs = np.array(optimized_wait_costs).reshape(
                *map_np.shape)
            optimized_edge_weights = comp_uncompress_edge_matrix(
                comp_map, weights[n_valid_vertices:])
            optimized_edge_weights = np.array(optimized_edge_weights).reshape(
                *map_np.shape, 4)
        else:
            optimized_wait_costs = np.zeros(map_np.shape)
            optimized_wait_costs[:] = weights[0]
            optimized_edge_weights = comp_uncompress_edge_matrix(
                comp_map, weights[1:])
            optimized_edge_weights = np.array(optimized_edge_weights).reshape(
                *map_np.shape, 4)

    # fig_edges, axs_edges = plt.subplots(2, 2, figsize=(15, 15))
    fig_edges = plt.figure(figsize=(25, 5))
    grid = AxesGrid(fig_edges,
                    111,
                    nrows_ncols=(1, 5),
                    axes_pad=0.5,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1)
    axs_edges = grid.axes_row
    ax_wait = axs_edges[0][4]
    # fig_wait, ax_wait = plt.subplots(1, 1, figsize=(8, 8))

    cmap = "Reds"
    # vmin = np.min([
    #     *optimized_edge_weights.flatten(), *optimized_wait_costs.flatten()
    # ]) * 1.2
    # vmax = np.max([
    #     *optimized_edge_weights.flatten(), *optimized_wait_costs.flatten()
    # ]) * 0.8
    vmin = 0
    vmax = 100

    cax1 = axs_edges[0][1].imshow(
        optimized_edge_weights[..., 0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axs_edges[0][1].set_title('Right', fontsize=30)
    # fig.colorbar(cax1, ax=axs[0,1])

    cax2 = axs_edges[0][2].imshow(
        optimized_edge_weights[..., 1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axs_edges[0][2].set_title('Up', fontsize=30)
    # fig.colorbar(cax2, ax=axs[1,0])

    cax3 = axs_edges[0][0].imshow(
        optimized_edge_weights[..., 2],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axs_edges[0][0].set_title('Left', fontsize=30)
    # fig.colorbar(cax3, ax=axs[0,0])

    cax4 = axs_edges[0][3].imshow(
        optimized_edge_weights[..., 3],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axs_edges[0][3].set_title('Down', fontsize=30)
    # fig.colorbar(cax4, ax=axs[1,1])

    if domain == "competition":
        ax_wait.imshow(
            optimized_wait_costs,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    elif domain == "kiva":
        ax_wait.imshow(
            optimized_wait_costs,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    ax_wait.set_title("Wait", fontsize=30)

    for i in range(len(axs_edges)):
        for j in range(len(axs_edges[0])):
            axs_edges[i][j].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            axs_edges[i][j].tick_params(
                axis='y',  # changes apply to the y-axis
                which='both',  # both major and minor ticks are affected
                left=False,  # ticks along the left edge are off
                right=False,  # ticks along the right edge are off
                labelleft=False)  # labels along the left edge are off

    cbar = axs_edges[0][4].cax.colorbar(cax4)
    cbar = grid.cbar_axes[0].colorbar(cax4)
    cbar.ax.tick_params(labelsize=25)

    # fig_edges.tight_layout()
    fig_edges.savefig(os.path.join(store_dir, f'{map_name}_highway.png'),
                      format='png',
                      bbox_inches='tight')
    fig_edges.savefig(os.path.join(store_dir, f'{map_name}_highway.pdf'),
                      format='pdf',
                      bbox_inches='tight')

    # ax_wait.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    # ax_wait.tick_params(
    #     axis='y',  # changes apply to the y-axis
    #     which='both',  # both major and minor ticks are affected
    #     left=False,  # ticks along the left edge are off
    #     right=False,  # ticks along the right edge are off
    #     labelleft=False)  # labels along the left edge are off
    # fig_wait.tight_layout()
    # fig_wait.savefig(os.path.join(store_dir, f'{map_name}_wait.png'),
    #                  format='png')
    # fig_wait.savefig(os.path.join(store_dir, f'{map_name}_wait.pdf'),
    #                  format='pdf')


def plot_highway_suggest_graph(map_np,
                               weights,
                               map_name,
                               comp_input_file=None,
                               store_dir=MAP_DIR,
                               domain="kiva"):
    """Plot highway suggest graph s.t:

    1. Vertices are the same as the original graph
    2. Edge direction between vertices u and v:
        a. Same as the edge with smaller edge cost between (u, v) and (v, u)
        b. “Suggest” direction for the robots
    3. Edge cost between vertices u and v:
        a. The color intensity indicates how “strong” the suggestion is.
        b. Computed by: abs(e(u, v) - e(v, u))
    """
    rows, cols = map_np.shape

    # Calculate the aspect ratio of the grid and figure size
    aspect_ratio = cols / rows
    user_figwidth = 18
    user_figsize = (user_figwidth, user_figwidth / aspect_ratio)

    # Derive scale factors from the figsize
    base_size = user_figwidth

    # Using 70% of the average space per node for the square
    square_size = base_size / max(rows, cols) * 0.7
    spacing_x = 2 * square_size * aspect_ratio
    spacing_y = 2 * square_size

    # Using 100% of the square size for the edge width
    edge_width = square_size

    # Create a new directed graph
    G = nx.DiGraph()

    # Add nodes with spacing derived from figsize and grid dimensions
    node_colors = []
    block_idxs = [
        kiva_obj_types.index("@"),
    ]
    # weights = map_json["weights"]
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c), pos=(spacing_x * c, -spacing_y * r))
            node_colors.append(kiva_color_map[map_np[r, c]])

    # Add directed edges with weights
    highway_s = kiva_get_highway_suggestion_graph(map_np, weights, block_idxs)
    for r in range(rows):
        for c in range(cols):
            # Skip all obstacles
            if map_np[r, c] in block_idxs:
                continue
            for idx, (dx, dy) in enumerate(kiva_directions):
                suggest_w = highway_s[r, c, idx]
                if not np.isnan(suggest_w):
                    G.add_edge((r, c), (r + dx, c + dy), weight=suggest_w)
            # if r > 0:
            #     G.add_edge((r, c), (r - 1, c), weight=np.random.rand())
            # if c > 0:
            #     G.add_edge((r, c), (r, c - 1), weight=np.random.rand())

    # Node and edge properties
    pos = nx.get_node_attributes(G, 'pos')
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    color_map = plt.cm.Reds
    norm = mcolors.Normalize(vmin=-0.4, vmax=MAX_WEIGHT)
    edge_colors = [
        color_map(norm(weight / MAX_WEIGHT)) for weight in edge_weights
    ]

    # Plotting with user-defined figure size
    fig, ax = plt.subplots(figsize=user_figsize)

    # Draw nodes as squares
    for node, (x, y) in pos.items():
        # Add black bounding box for empty space
        color = node_colors[node[0] * cols + node[1]]
        edge_color = 'black' if color == 'white' else color
        square = patches.Rectangle(
            (x - square_size / 2, y - square_size / 2),
            square_size,
            square_size,
            facecolor=color,
            edgecolor=edge_color,
        )
        ax.add_patch(square)

    # Draw edges with width derived from figsize
    nx.draw_networkx_edges(G,
                           pos,
                           width=edge_width,
                           edge_color=edge_colors,
                           arrowsize=35,
                           ax=ax)

    # Adjusting the limits of the axes to reduce whitespace
    ax.set_xlim(
        min(pos.values())[0] - square_size,
        max(pos.values())[0] + square_size)
    ax.set_ylim(
        min(pos.values(), key=lambda x: x[1])[1] - square_size,
        max(pos.values(), key=lambda x: x[1])[1] + square_size)

    # ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()

    fig.savefig(os.path.join(store_dir, f'{map_name}_highway_suggest.png'),
                format='png')
    fig.savefig(os.path.join(store_dir, f'{map_name}_highway_suggest.pdf'),
                format='pdf')


def main(map_filepath, store_dir=MAP_DIR, domain="kiva"):
    with open(map_filepath, "r") as f:
        map_json = json.load(f)
    if domain == "kiva":
        map_str, map_name = read_in_kiva_map(map_filepath)
        map_np = kiva_env_str2number(map_str)
    else:
        map_str, map_name = read_in_competition_map(map_filepath)
        map_np = competition_env_str2number(map_str)
    weights = map_json["weights"]
    optimize_wait = False
    if "optimize_wait" in map_json:
        optimize_wait = map_json["optimize_wait"]
    plot_highway_edge_weights(
        map_np,
        weights,
        map_name,
        map_filepath=map_filepath,
        store_dir=store_dir,
        domain=domain,
        optimize_wait=optimize_wait,
    )
    # plot_highway_suggest_graph(
    #     map_np,
    #     weights,
    #     map_name,
    #     map_filepath=map_filepath,
    #     store_dir=store_dir,
    #     domain=domain,
    # )


if __name__ == "__main__":
    fire.Fire(main)
