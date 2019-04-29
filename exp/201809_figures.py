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
                                           lims[particle], trajectories[particle][:, first_frame:last_frame], color[particle], vs=[50, 200, 250, 1300])

    plt.tight_layout()
    plt.savefig(str(img_save_dir), format='svg')
    plt.show()


df_dir = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\results\processed_non_cropped.pandas')
df = pd.read_pickle(str(df_dir))

sel_df = df.query('date == "20181109" and condition == "SAG_2" and cell == "cell_03"')
sel_df['foci_count'] = sel_df.area_labeled.apply(len)

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

control_cells = [('20181109', 'control', 'cell_01')]#,
                 #('20181109', 'control', 'cell_01'),
                 #('20190131', 'control', 'cell_01')]

control_cells_cropped = [('20190124', 'control', 'cell_01', 'cell_1.tif'),
                         ('20190124', 'control', 'cell_01', 'cell_2.tif')]#,
                         #('20190129', 'control', 'cell_04', 'cell_1.tif')]

decreasing_cells = [#('20181109', 'SAG', 'cell_03'),
                    ('20181109', 'SAG_2', 'cell_03'),
                    #('20181212', 'SAG_2', 'cell_01'),
                    #('20181212', 'SAG_2', 'cell_03'),
                    ('20181221', 'SAG_2', 'cell_01'),
                    ('20190118', 'SAG', 'cell_03'),
                    #reanalyze('20190124', 'SAG_2', 'cell_01'),
                    #reanalyze('20190124', 'SAG_2', 'cell_02'),
                    ('20190131', 'MET_2', 'cell_02'),
                    ('20190131', 'MET', 'cell_03')]#,
                    #('20190131', 'MET_2', 'cell_02')]

decreasing_cells_cropped = [#('20190118', 'SAG_2', 'cell_01', 'cell_2.tif'),
                            ('20190124', 'SAG', 'cell_02', 'cell_1.tif'),
                            ('20190129', 'SAG', 'cell_01', 'cell_1.tif'),
                            ('20190129', 'MET', 'cell_04', 'cell_1.tif'),
                            ('20190131', 'MET', 'cell_01', 'cell_2.tif'),
                            # no loose('20190131', 'MET', 'cell_02', 'cell_1.tif'),
                            ('20190131', 'MET', 'cell_02', 'cell_2.tif')] #,
                            # no loose ('20190131', 'MET_2', 'cell_03', 'cell_1.tif')]

all_cells = control_cells + decreasing_cells
all_cells_cropped = control_cells_cropped + decreasing_cells_cropped


def add_categories(df, column, separating_value, low_col_name=None, high_col_name=None):
    lower = []
    higher = []

    for vals in df[column].values:
        lower.append(np.sum(vals <= separating_value))
        higher.append(np.sum(vals > separating_value))

    if low_col_name is not None:
        df[low_col_name] = lower
    if high_col_name is not None:
        df[high_col_name] = higher

    return df


DATA_PATH = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\results')
non_cropped_dir = DATA_PATH.joinpath('processed_non_cropped.pandas')
cropped_dir = DATA_PATH.joinpath('processed_cropped.pandas')

df_non_crop = pd.read_pickle(str(non_cropped_dir))
df_crop = pd.read_pickle(str(cropped_dir))


# Correct delay in timings

for this_params, this_df in df_non_crop.groupby(['date', 'condition', 'cell']):
    print(this_params)
    if this_params[2] == 'cell_01':
        acq_times = this_df.query('time<=0').acquisition_date.values
        first_pos_acq_time = acq_times[-1]
        last_pre_acq_time = acq_times[-2]

        mid_point = first_pos_acq_time - last_pre_acq_time
        mid_point = mid_point / np.timedelta64(2, 'm')

        for i in this_df.index:
            df_non_crop.at[i, 'time'] = df_non_crop.time[i] + mid_point


for this_params, this_df in df_crop.groupby(['date', 'condition', 'cell']):
    print(this_params)
    if this_params[2] == 'cell_01':
        acq_times = this_df.query('time<=0').acquisition_date.values
        first_pos_acq_time = acq_times[-1]
        last_pre_acq_time = acq_times[-2]

        mid_point = first_pos_acq_time - last_pre_acq_time
        mid_point = mid_point / np.timedelta64(2, 'm')

        for i in this_df.index:
            df_crop.at[i, 'time'] = df_crop.time[i] + mid_point

for this_params, this_df in df_non_crop.groupby(['date', 'condition', 'cell']):
    print(this_params)
    if this_params[2] != 'cell_01':
        acq_time = this_df.query('time==0').acquisition_date.values[0]

        try:
            first_cell_acq_time = df_non_crop.query(
                'date == "%s" and condition == "%s" and cell == "cell_01" and time == 0' % (
                this_params[:2])).acquisition_date.values[0]
        except IndexError:

            print(this_params)
            print('no cell 01')

            acq_times = this_df.query('time<=0').acquisition_date.values
            first_pos_acq_time = acq_times[-1]
            last_pre_acq_time = acq_times[-2]

            mid_point = first_pos_acq_time - last_pre_acq_time
            mid_point = mid_point / np.timedelta64(2, 'm')

            for i in this_df.index:
                df_non_crop.at[i, 'time'] = df_non_crop.time[i] + mid_point

            continue

        time_diff = acq_time - first_cell_acq_time
        time_diff = time_diff / np.timedelta64(1, 'm')
        print(time_diff)

        for i in this_df.index:
            df_non_crop.at[i, 'time'] = df_non_crop.time[i] + time_diff

for this_params, this_df in df_crop.groupby(['date', 'condition', 'cell', 'time_step']):
    print(this_params)
    sel_df = df_non_crop.query(('date == "%s" and condition == "%s" and cell == "%s"' % this_params[:-1]))
    for i in this_df.index:
        this_sel_df = sel_df.query('acquisition_date == "%s"' % this_df.acquisition_date[i])
        new_time = this_sel_df.time.values[0]
        df_crop.at[i, 'time'] = new_time

sel_non_crop = []
for cell_param in all_cells:
    this_sel = df_non_crop.query('date == "%s" and condition == "%s" and cell == "%s"' % cell_param)
    sel_non_crop.append(this_sel)

sel_non_crop = pd.concat(sel_non_crop, ignore_index=True)
sel_non_crop['time_step'] = 'N/A'

sel_crop = []
for cell_param in all_cells_cropped:
    this_sel = df_crop.query('date == "%s" and condition == "%s" and cell == "%s" and time_step == "%s"' % cell_param)
    sel_crop.append(this_sel)

sel_crop = pd.concat(sel_crop, ignore_index=True)

all_df = pd.concat([sel_non_crop, sel_crop], ignore_index=True)

for i in all_df.query('date == "%s" and condition == "%s" and cell == "%s" and time_step == "%s"' % ('20181221', 'SAG_2', 'cell_01', 'N/A')).index:
    all_df.drop(i, inplace=True)

all_df = add_categories(all_df, 'area_labeled', 0.2, low_col_name='small', high_col_name='big')
all_df = add_categories(all_df, 'distances', 0.2, low_col_name='docked', high_col_name='loose')
all_df['total_granules'] = all_df.small + all_df.big


def normalize_column(df, column):
    df[column + '_normalized'] = np.nan
    for cell_params, this_df in df.groupby(['date', 'condition', 'cell', 'time_step']):
        pre_mean_val = np.mean(this_df.query('time < 0')[column].values)

        # if column == 'loose':
        #     pre_mean_total = np.mean(this_df.query('time < 0')['total_granules'].values)
        #     if pre_mean_val < .19 * pre_mean_total:
        #         continue

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
        plt.ylim((-0.01, 1.41))
        plt.title(column.split('_')[0])

    if savename is not None:
        img_save_dir = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8')
        img_save_dir = img_save_dir.joinpath(savename + '.svg')
        plt.savefig(str(img_save_dir), format='svg')


cols_to_normalize = ['small', 'big', 'docked', 'loose', 'total_granules']
for col in cols_to_normalize:
    all_df = normalize_column(all_df, col)

for col in cols_to_normalize:
    savename = col[:]
    col = col + '_normalized'
    plot_curves_and_mean(all_df, col, savename=savename)
    plt.close()


def append_time_binned_column(adf, steps, width):
    def func(value):
        if value < 0:
            return ' pre'
        for a in steps:
            if a <= value < a + width:
                return '%02d min' % a
        return 'very post'

    return adf.assign(moment=adf.time.apply(func))


def append_weighted_values(adf):
    adf['weight'] = np.nan
    cols = ['docked_normalized', 'loose_normalized']
    for col in cols:
        adf[col + '_weighted'] = np.nan
        params = ['date', 'condition', 'cell', 'time_step']
        for this_cell, this_cell_df in adf.groupby(params):
            pre_cell_df = this_cell_df.query('time < 0')
            weight = 1#pre_cell_df[col].values[0] / np.nanmean(pre_cell_df.total_granules.values[0]) # pre_cell_df[col].values[0] /

            for i in this_cell_df.index:
                adf.loc[i, 'weight'] = weight

        for this_moment, this_df in adf.groupby(['moment', 'drug']):
            weights_sum = np.nansum(this_df.weight.values * np.isfinite(this_df[col].values))

            for i in this_df.index:
                adf.loc[i, col + '_weighted'] = adf.at[i, col] * adf.loc[i, 'weight'] / weights_sum

    return adf


def generate_binned_df(df, columns):
    x = [0, 5, 10, 15, 20, 25, 30]
    end_x = [this + 20 for this in x]
    plot_dfs = []
    params = ['date', 'condition', 'cell', 'time_step']
    conditions = {'%02d min' % this_x: 'time > %s and time < %s' % (this_x, this_end_x)
                  for this_x, this_end_x in zip(x, end_x)}
    conditions[' pre'] = 'time < 0'
    for cell_params, this_df in df.groupby(params):
        for cond_name, condition in conditions.items():
            cell_dict = {param: [cell_param] for param, cell_param in zip(params, cell_params)}
            cell_df = pd.DataFrame(cell_dict)
            cell_df['moment'] = cond_name

            for column in columns:
                mini_df = this_df.query(condition)
                cell_df[column] = np.nanmean(mini_df[column].values)
                cell_df[column + '_std'] = np.nanstd(mini_df[column].values)
            plot_dfs.append(cell_df)

    return pd.concat(plot_dfs, ignore_index=True)

#binned = generate_binned_df(all_df, cols_to_normalize + [col + '_normalized' for col in cols_to_normalize])

bdf = append_time_binned_column(all_df, [0, 10, 20, 30], 10)
bdf['drug'] = bdf.condition.apply(lambda x: x.split('_')[0])
bdf = append_weighted_values(bdf)

bdf_excel_path = DATA_PATH.joinpath('all_decreasing_cells_binned.xls')
cols = ['total_granules', 'docked', 'loose']
cols += [col + '_normalized' for col in cols]
to_save = bdf[['date', 'condition', 'cell', 'time_step', 'time', 'moment', 'drug'] + cols]
to_save.to_excel(str(bdf_excel_path))


csv_path = DATA_PATH.joinpath('all_decreasing_cells.csv')
binned_csv_path = DATA_PATH.joinpath('all_decreasing_cells_binned.csv')

#to_save_df = all_df[['date', 'condition', 'cell', 'time_step', 'time', ] + cols_to_normalize + [col + '_normalized' for col in cols_to_normalize]]

#to_save_df.to_csv(str(csv_path))
#binned.to_csv(str(binned_csv_path))

# Save all cells separately
for cell_params, this_df in all_df.groupby(['date', 'condition', 'cell', 'time_step']):
    save_path = DATA_PATH
    for this_param in cell_params:
        if this_param == 'N/A':
            save_path = save_path.joinpath('NA')
        else:
            save_path = save_path.joinpath(this_param)
        save_path.mkdir(parents=True, exist_ok=True)

    save_path = save_path.joinpath('datos.xls')
    save_df = this_df[['time', 'docked', 'loose', 'total_granules', 'docked_normalized', 'loose_normalized', 'total_granules_normalized']]
    save_df.to_excel(str(save_path))


def plot_and_save_binned(df, col_a, col_b, savename):
    dockeds = df[['moment', col_a]].copy()
    dockeds.rename(columns={col_a: 'counts'}, inplace=True)
    dockeds['state'] = col_a.split('_')[0]
    looses = df[['moment', col_b]].copy()
    looses.rename(columns={col_b: 'counts'}, inplace=True)
    looses['state'] = col_b.split('_')[0]
    plot_df = pd.concat([dockeds, looses])

    sns.barplot(x='state', y='counts', hue='moment', data=plot_df, estimator=np.sum, ci=66)
    # sns.swarmplot(x='moment', y='counts', hue='state', data=df)
    # df.groupby('moment').plot(kind='bar', x='state', y='counts', yerr='')
    save_path = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8')
    save_path = save_path.joinpath(savename + '.svg')
    plt.savefig(str(save_path), format='svg')
    plt.close()

plot_and_save_binned(bdf, 'docked_normalized', 'loose_normalized', 'hist_docked')
plot_and_save_binned(bdf, 'small_normalized', 'big_normalized', 'hist_size')

all_means_dfs = []
for this_drug, this_df in bdf.groupby(['drug']):
    for this_moment, mini_df in this_df.groupby(['moment']):
        this_dict = {}

        for col in cols_to_normalize:

            mini_df['weight'] = np.nan
            params = ['date', 'condition', 'cell', 'time_step']
            for this_cell, this_cell_df in mini_df.groupby(params):
                filter_conditions = [this_param + ' == "' + this_cell_param + '"' for this_cell_param, this_param in
                                     zip(this_cell, params)]
                filter_condition = ' and '.join(filter_conditions)
                filter_condition += ' and moment == " pre"'
                pre_cell_df = bdf.query(filter_condition)
                weight = 1 #pre_cell_df[col].values[0] / pre_cell_df.total_granules.values[0]

                for i in this_cell_df.index:
                    mini_df.loc[i, 'weight'] = weight

            this_dict[col] = [np.nanmean(mini_df[col])]
            boots = bootstrap(mini_df[col])
            this_dict[col + '_bootstrap'] = [ci(boots, 66)]
            this_dict[col + '_normalized'] = [np.nansum(mini_df[col + '_normalized'].values * mini_df.weight.values) /
                                              np.nansum(mini_df.weight.values * np.isfinite(mini_df[col + '_normalized'].values))]
            this_dict[col + '_std'] = [np.nanstd(mini_df[col])]
            boots = bootstrap(mini_df[col + '_normalized'])
            this_dict[col + '_normalized_bootstrap'] = [ci(boots, 66)]
            this_dict[col + '_normalized_std'] = [np.nanstd(mini_df[col + '_normalized'])]
        this_dict['drug'] = this_drug
        this_dict['moment'] = this_moment
        all_means_dfs.append(pd.DataFrame(this_dict))

all_means_df = pd.concat(all_means_dfs, ignore_index=True)

to_save = all_means_df[['drug', 'moment'] + cols + [col + '_std' for col in cols] +
                       ['total_granules_bootstrap', 'docked_bootstrap', 'loose_bootstrap'] +
                       ['total_granules_normalized_bootstrap', 'docked_normalized_bootstrap', 'loose_normalized_bootstrap']]

to_save.to_excel(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8\data\means.xls')

drugs = ['SAG', 'MET', 'control']
for drug in drugs:
    this_df = all_means_df.query('drug == "%s"' % drug)
    plot_and_save_binned(this_df, 'docked_normalized', 'loose_normalized', 'hist_docked_' + drug)
    plot_and_save_binned(this_df, 'small_normalized', 'big_normalized', 'hist_size_' + drug)

for drug in drugs:
    this_df = bdf.query('drug == "%s"' % drug)
    plot_and_save_binned(this_df, 'docked_normalized_weighted', 'loose_normalized_weighted', 'barplot_' + drug)
    # plot_and_save_binned(this_df, 'small_normalized', 'big_normalized', 'barplot_size_' + drug)


for drug in drugs:
    this_df = bdf.query('drug == "%s"' % drug)
    plt.figure()
    for cell_params, this_cell in this_df.groupby(params):
        plt.plot(this_cell.docked.values, this_cell.loose.values, label=' '.join(cell_params))
        plt.scatter(this_cell.docked.values, this_cell.loose.values, c=this_cell.total_granules_normalized.values)
    plt.title(drug)
    plt.xlabel('docked')
    plt.ylabel('loose')
    plt.colorbar()
    plt.legend()
    plt.show()



# stat tests
import scipy.stats as st

def append_p_val(x, y, class_1, mom_1, class_2, mom_2, dictionary):
    mannwhitney = st.mannwhitneyu(x, y)

    dictionary['class_granule_1'].append(class_1)
    dictionary['moment_1'].append(mom_1)
    dictionary['class_granule_2'].append(class_2)
    dictionary['moment_2'].append(mom_2)
    dictionary['p_value'].append(mannwhitney.pvalue)

    return dictionary

kinds = ['docked', 'loose']

for this_drug, this_drug_df in bdf.groupby('drug'):
    print(this_drug)

    this_p_vals = {'class_granule_1': [],
               'moment_1': [],
               'class_granule_2': [],
               'moment_2': [],
               'p_value': []}


    for kind in kinds:

        x = this_drug_df.query('moment == " pre"')[kind + '_normalized'].values
        class_granule_1 = kind
        moment_1 = ' pre'
        class_granule_2 = kind

        for this_moment, this_df in this_drug_df.groupby('moment'):
            if this_moment == 'pre':
                continue

            y = this_df[kind + '_normalized'].values
            moment_2 = this_moment

            this_p_vals = append_p_val(x, y, class_granule_1, moment_1, class_granule_2, moment_2, this_p_vals)


    for this_moment, this_df in this_drug_df.groupby('moment'):

        x = this_df.docked_normalized.values
        y = this_df.loose_normalized.values

        class_granule_1 = 'docked'
        moment_1 = this_moment
        class_granule_2 = 'loose'
        moment_2 = this_moment

        this_p_vals = append_p_val(x, y, class_granule_1, moment_1, class_granule_2, moment_2, this_p_vals)

    for kind in kinds:

        x = this_drug_df.query('moment == "00 min"')[kind + '_normalized'].values
        y = this_drug_df.query('moment == "30 min"')[kind + '_normalized'].values

        class_granule_1 = kind
        moment_1 = '00 min'
        class_granule_2 = kind
        moment_2 = '30 min'

        this_p_vals = append_p_val(x, y, class_granule_1, moment_1, class_granule_2, moment_2, this_p_vals)

    p_values_df = pd.DataFrame(this_p_vals)
    p_values_df = p_values_df[['class_granule_1', 'moment_1', 'class_granule_2', 'moment_2', 'p_value']]

    p_values_df.to_excel(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8\data\p_values_' + this_drug + '.xls')

## Broken axis plot

f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', gridspec_kw={'width_ratios': [1, 3]})

for cell_params, this_df in all_df.groupby(['date', 'condition', 'cell', 'time_step']):
    ax.plot(list(this_df.query('time < 0').time.values) + [0, ],
            list(this_df.query('time < 0').total_granules_normalized.values) + [1, ],
            label=' '.join(cell_params))

    ax2.plot([0, ] + list(this_df.query('time >= 0').time.values),
             [1, ] + list(this_df.query('time >= 0').total_granules_normalized.values),
             label=' '.join(cell_params))

plt.legend(loc=1)
ax2.set_xlabel('time (min.)')
ax.set_ylabel('Normalized Quantity')
plt.title('Normalized Total Number of Granules')
# plot the same data on both axes


ax.set_xlim(-10,-5)
ax2.set_xlim(0,60)

# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelright='off')
ax2.yaxis.tick_right()

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

ax.plot((1-d,1+d), (-d,+d), **kwargs)
ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d/3,+d/3), (1-d,1+d), **kwargs)
ax2.plot((-d/3,+d/3), (-d,+d), **kwargs)

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'
plt.tight_layout()

save_path = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8')
save_path = save_path.joinpath('all_curves_plot_with_controls' + '.svg')
plt.savefig(str(save_path), format='svg')
plt.close()


#### borrar
for col in cols_to_normalize + [this_col + '_normalized' for this_col in cols_to_normalize]:
    f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', gridspec_kw={'width_ratios': [1, 3]})

    for cell_params, this_df in all_df.groupby(['date', 'condition', 'cell', 'time_step']):
        if cell_params[1] == 'control':
            color = 'b'
        elif cell_params[1] == 'SAG':
            color = 'r'
        elif cell_params[1] == 'MET':
            color = 'orange'

        ax.plot(list(this_df.query('time < 0').time.values) + [0, ],
                list(this_df.query('time < 0')[col].values) + [1, ],
                color,
                label=' '.join(cell_params))

        ax2.plot([0, ] + list(this_df.query('time >= 0').time.values),
                 [1, ] + list(this_df.query('time >= 0')[col].values),
                 color,
                 label=' '.join(cell_params))

    plt.legend(loc=1)
    ax2.set_xlabel('time (min.)')
    #ax.set_ylabel('Normalized Quantity')
    #plt.title('Normalized Total Number of Granules')
    # plot the same data on both axes


    ax.set_xlim(-10,-5)
    ax2.set_xlim(0,60)
    if 'normalized' in col:
        plt.ylim((0, 1.2))

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)

    ax.plot((1-d,1+d), (-d,+d), **kwargs)
    ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d/3,+d/3), (1-d,1+d), **kwargs)
    ax2.plot((-d/3,+d/3), (-d,+d), **kwargs)

    # What's cool about this is that now if we vary the distance between
    # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
    # the diagonal lines will move accordingly, and stay right at the tips
    # of the spines they are 'breaking'
    plt.tight_layout()

    save_path = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_8')
    save_path = save_path.joinpath(col + '.png')
    plt.savefig(str(save_path), format='png')
    plt.close()



# preguntas de tracking

all_parts = 0
all_docked = 0

for this_params, this_df in df.groupby(['date', 'condition', 'experiment', 'cell']):
    sel_df = this_df.query('frame == 0')
    this_particles = len(sel_df.distance.values)
    this_docked = np.sum(sel_df.distance.values < 0.3)

    all_parts += this_particles
    all_docked += this_docked
    print(this_params)
    print(this_particles, this_docked)


## Preguntas del video de la figura 1c

img_path = r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_1C\for_video_c01_001.tif'
imgs_save_dir = pathlib.Path(r'C:\Users\corba\Documents\Lab\s_granules\disgregation\unpublished\fig_1C')
img_save_dir = imgs_save_dir.joinpath('fig_vel_specific_video.svg')
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

df['time'] = df.frame.values * (469110)/((200 - 1) * 1000)
df.drop(147, inplace=True)

def add_velocity(df):
    df = df.sort_values('time')
    xs = df.X.values * 0.082
    ys = df.Y.values * 0.082
    times = df.time.values

    diff_xs = np.diff(xs)
    diff_ys = np.diff(ys)
    diff_times = np.diff(times)

    vels = np.sqrt(diff_xs ** 2 + diff_ys ** 2) / diff_times

    df.drop(df.index[0], inplace=True)

    df['velocity'] = vels

    return df

sel_df_a = df.query('particle == 2')
sel_df_a = add_velocity(sel_df_a)
sel_df_b = df.query('particle == 3 and time > 90')
sel_df_b = add_velocity(sel_df_b)
sel_df = pd.concat([sel_df_a, sel_df_b], ignore_index=True)
sel_df['particle'] = ['loose' if part == 2 else 'docked' for part in sel_df.particle]

sns.barplot(x='particle', y='velocity', data=sel_df)
plt.savefig(str(img_save_dir), format='svg')

import scipy.stats as st

print(st.ks_2samp(sel_df_a.velocity.values, sel_df_b.velocity.values))
print(st.mannwhitneyu(sel_df_a.velocity.values, sel_df_b.velocity.values))

# preguntas de all tracking

df = pd.read_pickle(r'C:\Users\corba\Documents\Lab\s_granules\all_tracking.pandas')

to_delete = [('20180505', 'cover_2_dsam', 'timelapse', 'c1_001.oif')]

for params in to_delete:
    sel_df = df.query('date == "%s" and condition == "%s" and experiment == "%s" and cell == "%s"'% params)
    for i in sel_df.index:
        df.drop(i, inplace=True)

docked_df = df.query('distance < 0.3')
loose_df = df.query('distance > 0.3')

def add_velocity_to_df(df):
    df['velocity'] = np.nan
    df['number'] = np.nan
    lengths = []
    durations = []
    count = 1
    for this_params, this_df in df.groupby(['date', 'condition', 'experiment', 'cell', 'particle']):
        print(this_params)
        this_df = this_df.sort_values('time')
        xs = this_df.X.values
        ys = this_df.Y.values
        zs = this_df.Z.values
        times = this_df.time.values

        diff_xs = np.diff(xs)
        diff_ys = np.diff(ys)
        if all(np.isfinite(zs)):
            diff_zs = np.diff(zs)
        else:
            diff_zs = 0
        diff_times = np.diff(times)

        lengths.append(len(diff_times))
        time_length = np.nansum(diff_times)
        if time_length > 0:
            durations.append(time_length)

        vels = np.sqrt(diff_xs ** 2 + diff_ys ** 2 + diff_zs ** 2) / diff_times

        for i, vel in zip(this_df.index, vels):
            df.at[i, 'velocity'] = vel
            df.at[i, 'number'] = count
        count += 1

    print(np.nanmean(lengths))
    print('mean duration: %f' % np.nanmean(durations))
    print('median duration: %f' % np.nanmedian(durations))
    print('percentile 25 duration: %f' % np.nanpercentile(durations, 25))
    print('percentile 75 duration: %f' % np.nanpercentile(durations, 75))
    return df

docked_df = add_velocity_to_df(docked_df)
loose_df = add_velocity_to_df(loose_df)

print('Analyzed %i docked particles' % (docked_df.number.max()))
print('Analyzed %i loose particles' % (loose_df.number.max()))

plot_df_a = docked_df[['time', 'velocity']]
plot_df_a['state'] = 'docked'
plot_df_b = loose_df[['time', 'velocity']]
plot_df_b['state'] = 'loose'
plot_df = pd.concat([plot_df_a, plot_df_b], ignore_index=True)

sns.barplot(x='state', y='velocity', data=plot_df)
#plt.savefig(str(img_save_dir), format='svg')

import scipy.stats as st

print(st.ks_2samp(plot_df_a.velocity.values, plot_df.velocity.values))
print(st.mannwhitneyu(plot_df_a.velocity.values, plot_df.velocity.values))
