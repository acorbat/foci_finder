import multiprocessing

import pandas as pd
import numpy as np

from foci_finder import foci_analysis as fa
from foci_finder import docking as dk
from foci_finder import tracking as tk


def get_x_step(oiffile):
    # TODO: check width is actually x
    return oiffile.mainfile['Reference Image Parameter']['WidthConvertValue']


def get_y_step(oiffile):
    # TODO: check height is actually y
    return oiffile.mainfile['Reference Image Parameter']['HeightConvertValue']


def get_z_step(oiffile):
    return oiffile.mainfile['Axis 3 Parameters Common']['Interval']


def my_iterator(N, foci_labeled, cell_segm, mito_segm):
    """Defined iterator to implement multiprocessing. Yields i in range and the segmented images given."""
    for i in range(N):
        yield i, foci_labeled, cell_segm, mito_segm

        
def evaluate_distance(foci_stack, mito_stack, path=None):
    # Find foci, cell and mitochondrias
    foci_labeled, cell_segm, mito_segm = fa.segment_all(foci_stack, mito_stack,
                                                        mito_filter_size=50,
                                                        mito_opening_disk=1)

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


def evaluate_superposition(foci_stack, mito_stack, N=500, path=None, max_dock_distance=3):
    """Pipeline that receives foci and mitocondrial stacks, segments foci, citoplasm and mitochondria. If path is given,
    segmentation is saved there. Superposition is evaluated and randomization of foci position is performed to evaluate
    correspondence with random positioning distribution. A DataFrame with calculated superpositions is returned."""
    # Find foci, cell and mitochondrias
    foci_labeled, cell_segm, mito_segm = fa.segment_all(foci_stack, mito_stack,
                                                        mito_filter_size=50,
                                                        mito_opening_disk=1)

    mito_segm = np.asarray([fa.binary_dilation(this, fa.disk(max_dock_distance)) for this in mito_segm])  # Dilate
    # mitochondria to see if foci are close but not superposed

    # Reorder foci, must try it
    foci_labeled = dk.relabel_by_area(foci_labeled)

    if path:
        fa.save_all(foci_labeled, cell_segm, mito_segm, path, axes='ZYX')

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
            cum_sim += np.asarray(rando_focis>0).astype(int)

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
        save_cum_dir = path.with_name(path.stem + '_cum_rand_foci.tif')
        fa.save_img(save_cum_dir, cum_sim, axes='ZYX')

    return res


def count_foci(foci_stack, mito_stack, path=None):
    """Pipeline that receives foci and mitocondrial stacks, segments foci, citoplasm and mitochondria. If path is given,
     segmentation is saved there. A DataFrame with foci found and their characterizations is returned."""
    foci_labeled, cell_segm, mito_segm = fa.segment_all(foci_stack, mito_stack,
                                                        mito_filter_size=50,
                                                        mito_opening_disk=1)

    if mito_segm is not None:
        df = fa.label_to_df(foci_labeled, cols=['label', 'centroid', 'coords', 'area', 'mean_intensity'],
                            intensity_image=mito_segm)
    else:
        df = fa.label_to_df(foci_labeled, cols=['label', 'centroid', 'coords', 'area'])

    if path:
        fa.save_all(foci_labeled, cell_segm, mito_segm, path)

    return df


def track_and_dock(foci_stack, mito_stack, dist_dock, path=None):
    foci_labeled, cell_segm, mito_segm = fa.segment_all(foci_stack, mito_stack, subcellular=True,
                                                        mito_filter_size=3,
                                                        mito_opening_disk=1)

    tracked = tk.track(foci_labeled, extra_attrs=['area', 'mean_intensity'], intensity_image=mito_segm)
    particle_labeled = tk.relabel_by_track(foci_labeled, tracked)

    if path:
        fa.save_all(particle_labeled, cell_segm, mito_segm, path)

    # Save image classifying docked foci
    tracked = tk.add_distances(tracked, particle_labeled, mito_segm)
    dock_vid = tk.relabel_video_by_dock(particle_labeled, tracked, dist_dock)
    save_dock_img_dir = path.with_name(path.stem + '_dock.tif')
    fa.save_img(save_dock_img_dir, dock_vid)

    tracked = tk.add_distances(tracked, particle_labeled, mito_segm, col_name='full_erode')

    return tracked
