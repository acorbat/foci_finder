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
from matplotlib.colors import LinearSegmentedColormap

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


def plot_zoom_cell_with_track_and_mito(grid, foci_img, mito_img, lims, track, color='r', vs=[50, 250, 200,1300]):
    ax = plt.subplot(grid)
    y, x = track
    vmin_foci, vmax_foci, vmin_mito, vmax_mito = vs
    ax.imshow(foci_img, cmap=cmap_foci, vmin=vmin_foci, vmax=vmax_foci)
    ax.imshow(mito_img, cmap=cmap_mito, vmin=vmin_mito, vmax=vmax_mito, alpha=0.8)
    ax.plot(x, y, c=color, lw=1)
    ax.set_ylim([lims[1], lims[0]])
    ax.set_xlim([lims[2], lims[3]])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def fig_foci_interacting(img_path=r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_1C\for_video_c01_001.tif'):
    imgs_save_dir = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_1C')
    img_save_dir = imgs_save_dir.joinpath('fig_1C.svg')
    img = tif.TiffFile(str(img_path))
    stack = img.asarray().astype(float)
    foci_stack = stack[:, 0]
    mito_stack = stack[:, 1]

    df, particle_labeled = track_stack(foci_stack)

    particles_to_merge = [2, 7, 8, 19]
    for i in df.index:
        if df.particle[i] in particles_to_merge:
            df.at[i, 'particle'] = 2

    particles = [2, 3]
    color = {2: '#1f77b4',
             3: 'orange'}

    cmap_mito = LinearSegmentedColormap.from_list('cmap_mito', ['black', (255/255, 4/255, 255/255)])
    cmap_mito.set_under('k', alpha=0)
    cmap_foci = LinearSegmentedColormap.from_list('cmap_foci', ['black', (22 / 255, 255 / 255, 22 / 255)])
    cmap_foci.set_under('k', alpha=1)

    trajectories = {}
    lims = {}
    for particle in particles:
        particle_df = df.query('particle == ' + str(particle))
        trajectories[particle] = np.asarray([particle_df.Y.values, particle_df.X.values])

        lims[particle] = (60, 127, 30, 127)
            # (np.clip(np.nanmin(trajectories[particle][0]) - 8, 0, stack.shape[0]),
            #               np.clip(np.nanmax(trajectories[particle][0]) + 8, 0, stack.shape[0]),
            #               np.clip(np.nanmin(trajectories[particle][1]) - 8, 0, stack.shape[0]),
            #               np.clip(np.nanmax(trajectories[particle][1]) + 8, 0, stack.shape[0]))

    #cell_borders = (0, 0, stack.shape[0]-1, stack.shape[-1]-1)

    fig = plt.figure(figsize=(6.4, 4))

    gs0 = gridspec.GridSpec(1, 2)
    #gs0.update(left=0.05, right=0.95)

    # Plot first frame
    first_frame = 39
    gs00 = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[0], wspace=0.05, hspace=0.05)

    # full image
    ax1 = plt.subplot(gs00[1:, :])
    ax1.imshow(foci_stack[first_frame], cmap=cmap_foci, vmin=50, vmax=300)
    ax1.imshow(mito_stack[first_frame], cmap=cmap_mito, vmin=250, vmax=1300, alpha=0.8)
    #ax1.set_ylim([cell_borders[1], cell_borders[0]])
    #ax1.set_xlim(cell_borders[2], cell_borders[3])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    # Add scalebar
    # 1um <-> 12.195122
    scalebar = AnchoredSizeBar(ax1.transData,
                               12, '          ', 'lower left',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=2,
                               label_top=True)  # ,
    # fontproperties=fontprops)

    ax1.add_artist(scalebar)

    # Add Timestamp
    # 1 frame <-> 2.35734
    plt.annotate('01:29', (5, 10), color='w', fontsize=14)

    dx_arrow = -10
    dy_arrow = 20
    for n, particle in enumerate(particles):
        # draw arrows
        ax1.arrow(trajectories[particle][1][first_frame] - dx_arrow + 5,
                  trajectories[particle][0][first_frame] - dy_arrow - 5,
                  dx=dx_arrow, dy=dy_arrow, head_width=10, head_length=15, fc=color[particle], ec=color[particle],
                  length_includes_head=True)

        # subplot tracks
        plot_zoom_cell_with_track_and_mito(gs00[0, n], foci_stack[first_frame], mito_stack[first_frame],
                                           lims[particle], trajectories[particle][:, :first_frame], color[particle])

    # Plot last frame
    last_frame = -1
    gs01 = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs0[1], wspace=0.05, hspace=0.05)

    ax4 = plt.subplot(gs01[1:, :])
    ax4.imshow(foci_stack[last_frame], cmap=cmap_foci, vmin=50, vmax=200)
    ax4.imshow(mito_stack[last_frame], cmap=cmap_mito, vmin=250, vmax=1300, alpha=0.8)
    #ax4.set_ylim([cell_borders[1], cell_borders[0]])
    #ax4.set_xlim(cell_borders[2], cell_borders[3])
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)


    # Add Timestamp
    # 1 frame <-> 2.35734
    plt.annotate('07:49', (5, 10), color='w', fontsize=14)

    for n, particle in enumerate(particles):
        # draw arrows
        ax4.arrow(trajectories[particle][1][last_frame] - dx_arrow + 5, trajectories[particle][0][last_frame] - dy_arrow - 5,
                  dx=dx_arrow, dy=dy_arrow, head_width=10, head_length=15, fc=color[particle], ec=color[particle],
                  length_includes_head=True)

        # subplot tracks
        plot_zoom_cell_with_track_and_mito(gs01[0, n], foci_stack[last_frame], mito_stack[last_frame],
                                           lims[particle], trajectories[particle], color[particle], vs=[50, 200, 250, 1300])

    plt.tight_layout()
    plt.savefig(str(img_save_dir), format='svg')
    plt.show()

df_dir = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\results\processed_non_cropped.pandas')
df = pd.read_pickle(str(df_dir))

sel_df = df.query('date == "20181109" and condition == "SAG_2" and cell == "cell_03"')
sel_df['foci_count'] = sel_df.area_labeled.apply(len)


def add_categories(df, column, separating_value, low_col_name=None, high_col_name=None):
    lower = []
    higher = []

    for vals in df[column].values:
        lower.append(np.sum(vals < separating_value))
        higher.append(np.sum(vals > separating_value))

    if low_col_name is not None:
        df[low_col_name] = lower
    if high_col_name is not None:
        df[high_col_name] = higher

    return df


smalls = []
bigs = []

for areas in sel_df.area_labeled.values:
    smalls.append(np.sum(areas < 0.2))
    bigs.append(np.sum(areas > 0.2))

sel_df['small_foci'] = smalls
sel_df['big_foci'] = bigs

plot_dict = {'moment': ['pre', '5 min', '10 min'] * 2,
             'count': bigs[1:4] + smalls[1:4],
             'size' : ['big'] *3 + ['small'] * 3}
plot_df = pd.DataFrame(plot_dict)

sns.barplot(x='size',  y='count', hue='moment', data=plot_df)
plt.savefig(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8\barplot.svg', format='svg')

sel_df = df.query('date == "20181109" and condition == "SAG_2" and cell == "cell_03"')
sel_df['foci_count'] = sel_df.distances.apply(len)

dockeds = []
looses = []

for distance in sel_df.distances.values:
    dockeds.append(np.sum(distance < 0.3))
    looses.append(np.sum(distance >= 0.3))

sel_df['docked'] = dockeds
sel_df['loose'] = looses

plot_dict = {'moment': ['pre', '5 min', '10 min'] * 2,
             'count': dockeds[1:4] + looses[1:4],
             'docked': ['docked'] *3 + ['loose'] * 3}
plot_df = pd.DataFrame(plot_dict)

sns.barplot(x='docked',  y='count', hue='moment', data=plot_df)
plt.savefig(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8\barplot_docked.svg', format='svg')

decreasing_cells = [('20181109', 'SAG', 'cell_03'),
                    ('20181109', 'SAG_2', 'cell_03'),
                    ('20181212', 'SAG_2', 'cell_01'),
                    ('20181212', 'SAG_2', 'cell_03'),
                    ('20181221', 'SAG_2', 'cell_01'),
                    ('20180118', 'SAG', 'cell_03'),
                    ('20180124', 'SAG_2', 'cell_01'),
                    ('20180124', 'SAG_2', 'cell_02'),
                    ('20180131', 'MET', 'cell_03'),
                    ('20180131', 'MET_2', 'cell_02')]

decreasing_cells_cropped = [('20180118', 'SAG_2', 'cell_01', 'cell_2.tif'),
                            ('20180124', 'SAG', 'cell_02', 'cell_1.tif'),
                            ('20180129', 'SAG', 'cell_01', 'cell_1.tif'),
                            ('20180131', 'MET_2', 'cell_03', 'cell_1.tif')]


DATA_PATH = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\results')
non_cropped_dir = DATA_PATH.joinpath('processed_non_cropped.pandas')
cropped_dir = DATA_PATH.joinpath('cropped.pandas')

df_non_crop = pd.read_pickle(str(non_cropped_dir))
df_crop = pd.read_pickle(str(cropped_dir))

sel_non_crop = []
for cell_param in decreasing_cells:
    this_sel = df_non_crop.query('date == "%s" and condition == "%s" and cell == "%s"' % cell_param)
    sel_non_crop.append(this_sel)

sel_non_crop = pd.concat(sel_non_crop, ignore_index=True)
sel_non_crop['time_step'] = 'cell_1.tiff'

sel_crop = []
for cell_param in decreasing_cells_cropped:
    this_sel = df_non_crop.query('date == "%s" and condition == "%s" and cell == "%s" and time_step == "%s"' % cell_param)
    sel_crop.append(this_sel)

sel_crop = pd.concat(sel_crop, ignore_index=True)

sel_non_crop = add_categories(sel_non_crop, 'area_labeled', 0.2, low_col_name='small', high_col_name='big')
sel_non_crop = add_categories(sel_non_crop, 'distances', 0.2, low_col_name='docked', high_col_name='loose')


def normalize_column(df, column):
    df[column + '_normalized'] = np.nan
    for cell_params, this_df in df.groupby(['date', 'condition', 'cell', 'time_step']):
        pre_mean_val = np.mean(this_df.query('time < 0')[column].values)

        if pre_mean_val == 0:
            continue

        for ind in this_df.index:
            df.at[ind, column + '_normalized'] = df.at[ind, column] / pre_mean_val

    return df


def plot_curves_and_mean(df, column, savename=None):
    time_cropped_df = df.query('time < 10.01')

    plt.figure(figsize=(3.2, 3.2))

    curves = []
    times = []
    for cell_params, this_df in time_cropped_df.groupby(['date', 'condition', 'cell', 'time_step']):
        this_curve = this_df[column].values
        this_times = this_df['time'].values

        plt.plot(this_times, this_curve, 'k', alpha=0.5)
        plt.xlabel('Time (min.)')
        plt.ylabel('Normalized Quantity')
        plt.title(column.split('_')[0])

    if savename is not None:
        img_save_dir = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8')
        img_save_dir = img_save_dir.joinpath(savename + '.svg')
        plt.savefig(str(img_save_dir), format='svg')


sel_non_crop = add_categories(sel_non_crop, 'area_labeled', 0.2, low_col_name='small', high_col_name='big')
sel_non_crop = add_categories(sel_non_crop, 'distances', 0.3, low_col_name='docked', high_col_name='loose')

cols_to_normalize = ['small', 'big', 'docked', 'loose']
for col in cols_to_normalize:
    sel_non_crop = normalize_column(sel_non_crop, col)

for col in cols_to_normalize:
    savename = col[:]
    col = col + '_normalized'
    plot_curves_and_mean(sel_non_crop, col, savename=savename)


def generate_binned_df(df, columns):
    plot_dfs = []
    params = ['date', 'condition', 'cell', 'time_step']
    conditions = {'pre': 'time < 0',
                  '5 min': 'time < 6 and time > -1',
                  '10 min': 'time < 11 and time > 6'}
    for cell_params, this_df in df.groupby(params):
        for cond_name, condition in conditions.items():
            cell_dict = {param: [cell_param] for param, cell_param in zip(params, cell_params)}
            cell_df = pd.DataFrame(cell_dict)
            cell_df['moment'] = cond_name

            for column in columns:
                mini_df = this_df.query(condition)
                cell_df[column] = np.nanmean(mini_df[column].values)
            plot_dfs.append(cell_df)

    return pd.concat(plot_dfs, ignore_index=True)


sns.barplot(x='docked', y='count', hue='moment', data=plot_df)
plt.savefig(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8\barplot_docked.svg',
            format='svg')
