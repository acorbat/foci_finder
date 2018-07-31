import multiprocessing

import pandas as pd
import numpy as np

from foci_finder import foci_analysis as fa
from foci_finder import docking as dk


def my_iterator(N, foci_labeled, cell_segm, mito_segm):
    """Defined iterator to implement multiprocessing. Yields i in range and the segmented images given."""
    for i in range(N):
        yield i, foci_labeled, cell_segm, mito_segm

        
def evaluate_distance(foci_stack, mito_stack, path=None):
    # Find foci, cell and mitochondrias
    foci_labeled, cell_segm, mito_segm = fa.segment_all(foci_stack, mito_stack)

    # Reorder foci, must try it
    foci_labeled = dk.relabel_by_area(foci_labeled)

    if path:
        fa.save_all(foci_labeled, cell_segm, mito_segm, path)

    # calculate pixel superposition
    exp_pix_sup = dk.calculate_superposition(foci_labeled, mito_segm)
    exp_foc_sup = dk.calculate_superposition(foci_labeled, mito_segm, 'label')
    
    distances = dk.calculate_distances(foci_labeled, mito_segm)
    
    distances['exp_pix_sup'] = exp_pix_sup
    distances['exp_foc_sup'] = exp_foc_sup
    
    return distances


def evaluate_superposition(foci_stack, mito_stack, N=500, path=None):
    """Pipeline that receives foci and mitocondrial stacks, segments foci, citoplasm and mitochondria. If path is given,
    segmentation is saved there. Superposition is evaluated and randomization of foci position is performed to evaluate
    correspondence with random positioning distribution. A DataFrame with calculated superpositions is returned."""
    # Find foci, cell and mitochondrias
    foci_labeled, cell_segm, mito_segm = fa.segment_all(foci_stack, mito_stack)

    # Reorder foci, must try it
    foci_labeled = dk.relabel_by_area(foci_labeled)

    if path:
        fa.save_all(foci_labeled, cell_segm, mito_segm, path)

    # calculate pixel superposition
    exp_pix_sup = dk.calculate_superposition(foci_labeled, mito_segm)
    exp_foc_sup = dk.calculate_superposition(foci_labeled, mito_segm, 'label')

    # randomize N times foci location to estimate random superposition percentage
    output = dict()
    with multiprocessing.Pool(31) as p:

        cum_sim = np.zeros_like(foci_labeled)

        for i, superpositions, rando_focis in p.imap_unordered(dk.randomize_and_calculate,
                                                  my_iterator(N, foci_labeled, cell_segm, mito_segm)):
            print('Performing iteration %d' % i)
            output[i] = superpositions
            cum_sim += rando_focis

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

        # Save cumulative randomization foci position
        fa.save_img(path, cum_sim)

    return res


def count_foci(foci_stack, mito_stack, path=None):
    """Pipeline that receives foci and mitocondrial stacks, segments foci, citoplasm and mitochondria. If path is given,
     segmentation is saved there. A DataFrame with foci found and their characterizations is returned."""
    foci_labeled, cell_segm, mito_segm = fa.segment_all(foci_stack, mito_stack)

    if mito_segm is not None:
        df = fa.label_to_df(foci_labeled, cols=['label', 'centroid', 'coords', 'area', 'mean_intensity'],
                            intensity_image=mito_segm)
    else:
        df = fa.label_to_df(foci_labeled, cols=['label', 'centroid', 'coords', 'area'])

    if path:
        dk.save_all(foci_labeled, cell_segm, mito_segm, path)

    return df


def track_and_dock(foci_stack, mito_stack, path=None):
    foci_labeled, cell_segm, mito_segm = fa.segment_all(foci_stack, mito_stack, subcellular=True)

    tracked = dk.track(foci_labeled, extra_attrs=['area', 'mean_intensity'], intensity_image=mito_segm)
    particle_labeled = dk.relabel_by_track(foci_labeled, tracked)

    if path:
        fa.save_all(particle_labeled, cell_segm, mito_segm, path)

    tracked = dk.add_distances(tracked, particle_labeled, mito_segm)
    tracked = dk.add_distances(tracked, particle_labeled, mito_segm, col_name='full_erode')

    return tracked
