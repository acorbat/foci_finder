import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages

from img_manager import oiffile as oif
from img_manager import tifffile as tif

from foci_finder import foci_analysis as fa
from foci_finder import tracking as tk

matplotlib.rcParams.update({'font.size': 8})

def load_img(filename):
    imgs_path = pathlib.Path('/mnt/data/Laboratorio/uVesiculas/docking/')
    img_path = imgs_path.joinpath(filename)
    img = oif.OifFile(str(img_path))
    return img.asarray()[0].astype(float)


def load_df(filename):
    dfs_path = pathlib.Path('/mnt/data/Laboratorio/uVesiculas/docking/')
    df_path = dfs_path.joinpath(filename)
    return pd.read_pickle(str(df_path))


def track_stack(stack):
    foci_labeled = np.zeros_like(stack, dtype=int)
    for t, this_foci_stack in enumerate(stack):
        foci_labeled[t] = fa.find_foci(this_foci_stack, LoG_size=[2, 2])
    tracked = tk.track(foci_labeled, max_dist=20, gap=1, extra_attrs=['area'], intensity_image=[None]*len(stack),
                       subtr_drift=False)

    particle_labeled = tk.relabel_by_track(foci_labeled, tracked)

    return tracked, particle_labeled


def plot_zoom_cell_with_track(grid, img, lims, track, color='r'):
    ax = plt.subplot(grid)
    y, x = track
    ax.imshow(img)
    ax.plot(x, y, c=color, lw=1)
    ax.set_ylim([lims[1], lims[0]])
    ax.set_xlim([lims[2], lims[3]])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def fig_cell_tracked(img_path='20180503/cover_3/c1_con_foci_001.oif'):
    imgs_save_dir = pathlib.Path('/mnt/data/Laboratorio/uVesiculas/201809_figures/')
    img_save_dir = imgs_save_dir.joinpath('fig_1/fig_1.svg')
    stack = load_img(img_path)

    df, particle_labeled = track_stack(stack)

    particles = [10, 24, 3]
    color = {10: 'g',
             24: 'y',
             3: 'orange'}

    trajectories = {}
    lims = {}
    for particle in particles:
        particle_df = df.query('particle == ' + str(particle))
        trajectories[particle] = [particle_df.Y.values, particle_df.X.values]

        lims[particle] = (np.nanmin(trajectories[particle][0] - 8),
                          np.nanmin(trajectories[particle][0] + 18),
                          np.nanmin(trajectories[particle][1] - 8),
                          np.nanmin(trajectories[particle][1] + 18))

    cell_borders = (50, 480, 0, stack.shape[-1]-1)

    fig = plt.figure(figsize=(6.4, 4))

    gs0 = gridspec.GridSpec(1, 2)
    #gs0.update(left=0.05, right=0.95)

    # Plot first frame
    gs00 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[0], wspace=0.05, hspace=0.0)

    # full image
    ax1 = plt.subplot(gs00[1:, :])
    ax1.imshow(stack[0])
    ax1.set_ylim([cell_borders[1], cell_borders[0]])
    ax1.set_xlim(cell_borders[2], cell_borders[3])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    dx_arrow = -10
    dy_arrow = 20
    for n, particle in enumerate(particles):
        # draw arrows
        ax1.arrow(trajectories[particle][1][0] - dx_arrow + 5, trajectories[particle][0][0] - dy_arrow - 5, dx=dx_arrow,
                  dy=dy_arrow, head_width=10, head_length=15, fc=color[particle], ec=color[particle],
                  length_includes_head=True)

        # subplot tracks
        plot_zoom_cell_with_track(gs00[0, n], stack[0], lims[particle], trajectories[particle], color[particle])

    # Plot last frame
    gs01 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[1], wspace=0.05, hspace=0.0)

    ax4 = plt.subplot(gs01[1:, :])
    ax4.imshow(stack[-1])
    ax4.set_ylim([cell_borders[1], cell_borders[0]])
    ax4.set_xlim(cell_borders[2], cell_borders[3])
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    for n, particle in enumerate(particles):
        # draw arrows
        ax4.arrow(trajectories[particle][1][-1] - dx_arrow + 5, trajectories[particle][0][-1] - dy_arrow - 5,
                  dx=dx_arrow, dy=dy_arrow, head_width=10, head_length=15, fc=color[particle], ec=color[particle],
                  length_includes_head=True)

        # subplot tracks
        plot_zoom_cell_with_track(gs01[0, n], stack[-1], lims[particle], trajectories[particle], color[particle])

    plt.tight_layout()
    plt.savefig(str(img_save_dir), format='svg')
    plt.show()
