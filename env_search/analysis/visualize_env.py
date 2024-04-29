import os
import fire
from typing import List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import cv2
import imageio
import moviepy.editor as mp

from PIL import Image
from scipy.sparse import csgraph
from functools import partial
from matplotlib import colors

from env_search import MAP_DIR
from env_search.utils import (
    kiva_obj_types,
    kiva_env_str2number,
    kiva_env_number2str,
    read_in_kiva_map,
    KIVA_ROBOT_BLOCK_WIDTH,
    read_in_competition_map,
    competition_env_str2number,
)
from env_search.utils import set_spines_visible

FIG_HEIGHT = 10


def convert_avi_to_gif(
        input_path,
        output_path,
        output_resolution=(640, 640),
):
    # Load the input video file
    clip = mp.VideoFileClip(input_path)

    # Convert the video to a sequence of frames
    frames = []
    for frame in clip.iter_frames():
        # Resize the frame to the desired output resolution
        frame = Image.fromarray(frame).resize(output_resolution)
        frames.append(frame)

    # Write the frames to a GIF file
    # imageio.mimsave(output_path, frames, fps=clip.fps, size=output_resolution)
    imageio.mimsave(
        output_path,
        frames,
        fps=clip.fps,
        format='gif',
        palettesize=256,
    )


def create_movie(folder_path, filename):
    glob_str = os.path.join(folder_path, '*.png')
    image_files = sorted(glob.glob(glob_str))

    # Grab the dimensions of the image
    img = cv2.imread(image_files[0])
    image_dims = img.shape[:2][::-1]

    # Create a video
    avi_output_path = os.path.join(folder_path, f"{filename}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_rate = 30
    video = cv2.VideoWriter(
        avi_output_path,
        fourcc,
        frame_rate,
        image_dims,
    )

    for img_filename in image_files:
        img = cv2.imread(img_filename)
        video.write(img)

    video.release()

    # Convert video to gif
    gif_output_path = os.path.join(folder_path, f"{filename}.gif")
    convert_avi_to_gif(avi_output_path, gif_output_path, image_dims)


def visualize_env(env_np, cmap, norm, ax, fig, save, filenames, store_dir,
                  dpi):
    # heatmap = plt.pcolor(np.array(data), cmap=cmap, norm=norm)
    # plt.colorbar(heatmap, ticks=[0, 1, 2, 3])
    sns.heatmap(
        env_np,
        square=True,
        cmap=cmap,
        norm=norm,
        ax=ax,
        cbar=False,
        rasterized=True,
        annot_kws={"size": 30},
        linewidths=1,
        linecolor="black",
        xticklabels=False,
        yticklabels=False,
    )

    set_spines_visible(ax)
    ax.figure.tight_layout()

    if save:
        ax.margins(x=0, y=0)
        for filename in filenames:
            fig.savefig(
                os.path.join(store_dir, filename),
                dpi=dpi,
                bbox_inches='tight',
                # pad_inches=0,
                rasterized=True,
            )
        plt.close('all')


def visualize_competition(
    env_np: np.ndarray,
    filenames: List = None,
    store_dir: str = MAP_DIR,
    dpi: int = 300,
    ax: plt.Axes = None,
    figsize: tuple = None,
):
    visualize_kiva(env_np, filenames, store_dir, dpi, ax, figsize)


def visualize_kiva(
    env_np: np.ndarray,
    filenames: List = None,
    store_dir: str = MAP_DIR,
    dpi: int = 300,
    ax: plt.Axes = None,
    figsize: tuple = None,
):
    """
    Visualize kiva layout. Will store image under `store_dir`

    Args:
        env_np: layout in numpy format
    """
    n_row, n_col = env_np.shape
    save = False
    if ax is None:
        if figsize is None:
            # figsize = (n_col, n_row)
            figsize = (FIG_HEIGHT * n_col / n_row, FIG_HEIGHT)
            # if n_col > 50 or n_row > 50:
            #     figsize = (16, 16)
        # fig, ax = plt.subplots(1, 1, figsize=(n_col, n_row))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        save = True
    else:
        fig = ax.get_figure()
    cmap = colors.ListedColormap(
        ['white', 'black', 'deepskyblue', 'orange', 'fuchsia'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    visualize_env(env_np, cmap, norm, ax, fig, save, filenames, store_dir, dpi)


def main(map_filepath, store_dir=MAP_DIR, domain="kiva"):
    """
    Args:
        domain: one of ['kiva', 'competition']
    """
    if domain == "kiva":
        kiva_map, map_name = read_in_kiva_map(map_filepath)
        visualize_kiva(kiva_env_str2number(kiva_map),
                       store_dir=store_dir,
                       filenames=[f"{map_name}.png"])
    elif domain == "competition":
        # For competition map, `map_filepath` is competition input file
        competition_map, map_name = read_in_competition_map(map_filepath)
        # The visualization is the same as kiva because competition maps only
        # have empty spaces and obstacles
        visualize_competition(competition_env_str2number(competition_map),
                              store_dir=store_dir,
                              filenames=[f"{map_name}.png"])


if __name__ == '__main__':
    fire.Fire(main)
