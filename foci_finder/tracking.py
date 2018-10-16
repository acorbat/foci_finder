from foci_finder import foci_analysis as fa
from foci_finder import docking as dk
from tracker.track import *


def relabel_by_track(labeled_mask, track_df):
    """Relabels according to particle column in track_df every frame of labeled_mask according to track_df."""
    out = np.zeros_like(labeled_mask)
    for frame, df in track_df.groupby('frame'):
        swap = [[df.particle[i], df.label[i]] for i in df.index]

        out[frame] = fa.relabel(labeled_mask[frame], swap)

    return out


def add_distances(tracked, particle_labeled, mito_segm, col_name='distance'):
    distances = []
    for i in tracked.index:
        t = tracked.frame[i]
        particle = tracked.particle[i]

        print('Analyzing particle %d in frame %d' % (particle, t))

        focus_mask = particle_labeled[t] == particle
        mito_segm_sel = mito_segm[t]

        dist = dk.evaluate_distance(focus_mask, mito_segm_sel)
        distances.append(dist)
        print(dist)
    tracked[col_name] = distances

    return tracked


def relabel_video_by_dock(labeled_stack, df, cond, col='distance'):
    """Takes a time series of labeled stacks, its df and paints each foci in each time stack according to its docking
    state."""
    new_stack = np.zeros_like(labeled_stack)
    for t, stack in enumerate(labeled_stack):
        new_stack[t] = dk.relabel_by_dock(stack, df.query('frame == ' + str(t)), cond, col=col)

    return new_stack
