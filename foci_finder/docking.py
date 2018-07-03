import multiprocessing

import pandas as pd
import numpy as np
import numba as nb

from skimage.measure import label, regionprops

from foci_finder import foci_analysis as fa


def move_focus(focus_coord, new_position):
    """Takes first element to new position, translating the whole set of coordinates with it."""
    return focus_coord - focus_coord[0] + new_position


@nb.njit
def mycheck(cell, focus):
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

    :param cell_mask: binary array, M, N, O
    :param foci_mask: list of binary array of Mi, Mi, Oi
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


def randomize_foci_positions(foci_df, cell_coords):
    """Takes a foci DataFrame and randomizes the positions of foci into cell_coords."""
    random_foci = foci_df.copy()
    new_poss = np.random.choice(len(cell_coords), size=len(random_foci), replace=False)
    new_coords = [move_focus(foci_coord, cell_coords[new_pos]) for foci_coord, new_pos in
                  zip(random_foci.coords.values, new_poss)]
    random_foci['coords'] = new_coords

    return random_foci


def reconstruct_label_from_df(df, shape, extra_stack=5):
    """Takes a DataFrame and creates a labeled imaged of shape with the coordinates in df.coords"""
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
        # Todo: Test this function
        n_focis = 0
        n_focis_sup = 0
        for region in regionprops(foci_labeled, intensity_image=mito_segm):
            n_focis += 1
            n_focis_sup += region.max_intensity

        return n_focis_sup/n_focis

    else:
        raise LookupError


def segment_all(foci_stack, mito_stack):
    foci_labeled = fa.find_foci(foci_stack)
    cell_segm = fa.find_cell(foci_stack, foci_labeled > 0)
    if mito_stack:
        mito_segm = fa.find_mito(mito_stack, cell_segm, foci_labeled > 0)
    else:
        mito_segm = None

    return foci_labeled, cell_segm, mito_segm


def randomize_and_calculate(params):
    i, foci_labeled, cell_segm, mito_segm = params
    new_focis = rando(foci_labeled, cell_segm)
    new_focis = label(new_focis)

    superpositions = [calculate_superposition(new_focis, mito_segm, how=key) for key in ['pixel', 'label']]

    return i, superpositions


def my_iterator(N, foci_labeled, cell_segm, mito_segm):
    for i in range(N):
        yield i, foci_labeled, cell_segm, mito_segm


def evaluate_superposition(foci_stack, mito_stack, N=500):
    # Find foci, cell and mitochondrias
    foci_labeled, cell_segm, mito_segm = segment_all(foci_stack, mito_stack)

    # calculate pixel superposition
    exp_pix_sup = calculate_superposition(foci_labeled, mito_segm)
    exp_foc_sup = calculate_superposition(foci_labeled, mito_segm, 'label')

    # randomize N times foci location to estimate random superposition percentage
    output = dict()
    with multiprocessing.Pool(12) as p:
        for i, superpositions in p.imap_unordered(randomize_and_calculate, my_iterator(N, foci_labeled, cell_segm, mito_segm)):
            print(i)
            output[i] = superpositions

        superpositions = {'pixel': [], 'label': []}
        for vals in output.values():
            for j, key in enumerate(['pixel', 'label']):
                superpositions[key].append(vals[j])

        res = {'n_foci': [foci_labeled.max()],
               'experimental_pixel_superposition': [exp_pix_sup],
               'experimental_foci_superposition': [exp_foc_sup],
               'randomized_pixel_superposition': [superpositions['pixel']],
               'randomized_foci_superposition': [superpositions['label']]}
        res = pd.DataFrame.from_dict(res)

    return res


def count_foci(foci_stack, mito_stack, path=None):
    foci_labeled, cell_segm, mito_segm = segment_all(foci_stack, mito_stack)

    df = fa.label_to_df(foci_labeled, cols=['label', 'centroid', 'coords', 'area'])

    if path:
        foci_path = path.with_name(path.stem + '_foci_segm.tiff')
        fa.save_img(foci_path, foci_labeled)
        cell_path = path.with_name(path.stem + '_cell_segm.tiff')
        fa.save_img(cell_path, cell_segm)
        if mito_segm:
            mito_path = path.with_name(path.stem + '_mito_segm.tiff')
            fa.save_img(mito_path, mito_segm)

    return df
