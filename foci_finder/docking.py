import pandas as pd
import numpy as np
import numba as nb
import trackpy as tp

from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation, disk

from foci_finder import foci_analysis as fa


def move_focus(focus_coord, new_position):
    """Takes first element to new position, translating the whole set of coordinates with it."""
    return focus_coord - focus_coord[0] + new_position


@nb.njit
def mycheck(cell, focus):
    """Checks possible positions for the specific focus inside the cell volume. Uses numba for optimization."""
    out = np.zeros_like(cell)
    M, N, O = cell.shape
    P, Q, R = focus.shape
    sfocus = np.sum(focus)
    for m in range(M-P):
        for n in range(N-Q):
            for o in range(O-R):
                if np.sum(cell[m:(m+P), n:(n+Q), o:(o+R)] * focus) == sfocus:
                    out[m, n, o] = 1

    return out


def rando(foci_label, cell_mask):
    """
    Takes the labeled foci stack, and iterates over each foci repositioning them inside the cell segmentation volume.

    :param cell_mask: binary array, M, N, O
    :param foci_label: list of binary array of Mi, Mi, Oi
    :return: binary array, M, N, O
    """
    new_foci = np.zeros_like(cell_mask)
    free = np.copy(cell_mask)
    for region in regionprops(foci_label):
        focus_bbox = region.bbox
        focus_mask = foci_label[focus_bbox[0]:focus_bbox[3], focus_bbox[1]:focus_bbox[4], focus_bbox[2]:focus_bbox[5]]
        focus_mask = focus_mask > 0
        P, Q, R = focus_mask.shape
        m, n, o = np.nonzero(mycheck(free, focus_mask))
        if not len(m):
            raise Exception("Exception")

        ndx = np.random.choice(range(len(m)))
        m, n, o = m[ndx], n[ndx], o[ndx]
        new_foci[m:(m+P), n:(n+Q), o:(o+R)] = focus_mask
        free[m:(m+P), n:(n+Q), o:(o+R)] -= focus_mask

    return new_foci


def relabel_by_area(labeled_mask, reverse=True):
    """Takes a labeled foci images and relabels it according to volume size. If reverse=True largest foci is smallest
    label."""
    areas = sorted(((np.sum(labeled_mask == ndx), ndx) for ndx in np.unique(labeled_mask.flatten()) if ndx > 0),
                   reverse=reverse)

    swap = list()
    for new, (area, old) in enumerate(areas, 1):
        swap.append([old, new])

    out = fa.relabel(labeled_mask, swap)

    return out


def relabel_by_track(labeled_mask, track_df):
    """Relabels according to particle column in track_df every frame of labeled_mask according to track_df."""
    out = np.zeros_like(labeled_mask)
    for frame, df in track_df.groupby('frame'):
        swap = [[df.particle[i], df.label[i]] for i in df.index]

        out[frame] = fa.relabel(labeled_mask[frame], swap)

    return out


def track(labeled_stack, extra_attrs=None, intensity_image=None):
    """Takes labeled_stack of time lapse, prepares a DataFrame from the labeled images, saving centroid positions to be
    used in tracking by trackpy. extra_attrs is a list of other attributes to be saved into the tracked dataframe."""
    elements = []
    for t, stack in enumerate(labeled_stack):  # save each desired attribute of each label from each stack into a dict.
        for region in regionprops(stack, intensity_image=intensity_image[t]):
            element = {'frame': t, 'label': region.label}

            centroid = {axis: pos for axis, pos in zip(['x', 'y', 'z'], reversed(region.centroid))}
            element.update(centroid)

            if extra_attrs is not None:
                extra = {attr: region[attr] for attr in extra_attrs}
                element.update(extra)

            elements.append(element)
    elements = pd.DataFrame(elements)
    elements = tp.link_df(elements, 15)
    elements['particle'] += 1

    return elements


def evaluate_distance(focus_mask, mito_segm):
    n = 0
    if len(focus_mask.shape) == 3:
        def my_erode(focus_mask):
            return np.asarray([binary_erosion(this_focus_mask) for this_focus_mask in focus_mask])

        def my_dilate(focus_mask):
            return np.asarray([binary_dilation(this_focus_mask) for this_focus_mask in focus_mask])

    elif len(focus_mask.shape) == 2:
        def my_erode(focus_mask):
            return binary_erosion(focus_mask)

        def my_dilate(focus_mask):
            return binary_dilation(focus_mask)

    else:
        raise ValueError('dimension mismatch')

    if any(mito_segm.flatten()):
        if any(mito_segm[focus_mask]):
            while any(mito_segm[focus_mask]):
                n += 1
                focus_mask = my_erode(focus_mask)

            return -n

        else:
            while not any(mito_segm[focus_mask]):
                n += 1
                focus_mask = my_dilate(focus_mask)

            return n

    else:
        return np.nan


def add_distances(tracked, particle_labeled, mito_segm, col_name='distance'):
    distances = []
    for i in tracked.index:
        t = tracked.frame[i]
        particle = tracked.particle[i]

        print('Analyzing particle %d in frame %d' % (particle, t))

        focus_mask = particle_labeled[t] == particle
        mito_segm_sel = mito_segm[t]

        dist = evaluate_distance(focus_mask, mito_segm_sel)
        distances.append(dist)
        print(dist)
    tracked[col_name] = distances

    return tracked


def randomize_foci_positions(foci_df, cell_coords):
    """(deprecated) Takes a foci DataFrame and randomizes the positions of foci into cell_coords.
    Used to be the position randomization function."""
    random_foci = foci_df.copy()
    new_poss = np.random.choice(len(cell_coords), size=len(random_foci), replace=False)
    new_coords = [move_focus(foci_coord, cell_coords[new_pos]) for foci_coord, new_pos in
                  zip(random_foci.coords.values, new_poss)]
    random_foci['coords'] = new_coords

    return random_foci


def reconstruct_label_from_df(df, shape, extra_stack=5):
    """(deprecated, mught be necessary later) Takes a DataFrame and creates a labeled imaged of shape with the
    coordinates in df.coords"""
    rec = np.zeros(shape)
    rec = np.concatenate([rec, np.zeros((extra_stack,) + rec.shape[1:])])
    for i in df.index:
        rec[tuple(df.coords[i].T)] = df.label[i]
    return rec[:-extra_stack]


def calculate_superposition(foci_labeled, mito_segm, how='pixel'):
    """Takes the labeled foci stack and segmented mitochondria stack and check percentage of pixel/labels
    superposition."""
    if how == 'pixel':
        mito_sup = mito_segm[foci_labeled > 0]
        return np.sum(mito_sup) / len(mito_sup)

    elif how == 'label':
        n_focis = 0
        n_focis_sup = 0
        for region in regionprops(foci_labeled, intensity_image=mito_segm):
            n_focis += 1
            n_focis_sup += region.max_intensity

        return n_focis_sup/n_focis

    else:
        raise LookupError


def randomize_and_calculate(params):
    """Takes an index i for iteration number, and segmentation stacks. Foci are realocated using rando function into
    cell segm and then superposition is calculated. Index and superpositions is returned."""
    i, foci_labeled, cell_segm, mito_segm = params
    cell_mask = np.ma.array(cell_segm)
    foci_mask = np.ma.array(foci_labeled > 0)
    mask = np.ma.array(np.ones(foci_labeled.shape), mask=(cell_mask + foci_mask))
    mask = np.asarray([binary_dilation(this.mask, selem=disk(2)) for this in mask])

    new_focis = rando(foci_labeled, mask)
    new_focis = label(new_focis)

    superpositions = [calculate_superposition(new_focis, mito_segm, how=key) for key in ['pixel', 'label']]

    return i, superpositions
