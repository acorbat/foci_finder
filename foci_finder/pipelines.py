import pandas as pd
import numpy as np
import multiprocessing

from . import foci_analysis as fa
from . import docking as dk
from . import tracking as tk


def get_x_step(oiffile):
    return oiffile.mainfile['Reference Image Parameter']['WidthConvertValue']


def get_y_step(oiffile):
    return oiffile.mainfile['Reference Image Parameter']['HeightConvertValue']


def get_z_step(oiffile):
    return oiffile.mainfile['Axis 3 Parameters Common']['Interval'] / 1000  # It was in nanometers


def get_t_step(oiffile):
    start = oiffile.mainfile['Axis 4 Parameters Common']['StartPosition']
    end = oiffile.mainfile['Axis 4 Parameters Common']['EndPosition']
    size_t = oiffile.mainfile['Axis 4 Parameters Common']['MaxSize']
    return (end - start) / ((size_t - 1) * 1000)


def get_axis(oiffile):
    axes = oiffile.mainfile['Axis Parameter Common']['AxisOrder']
    axes = axes[2:] + 'YX'
    return axes


def get_clip_bbox(oiffile):
    x_start = oiffile.mainfile['Axis 0 Parameters Common']['ClipPosition']
    y_start = oiffile.mainfile['Axis 1 Parameters Common']['ClipPosition']
    x_size = oiffile.mainfile['Axis 0 Parameters Common']['MaxSize']
    y_size = oiffile.mainfile['Axis 1 Parameters Common']['MaxSize']
    return (y_start, y_start + y_size, x_start, x_start + x_size)


def my_iterator(N, foci_labeled, cell_segm, mito_segm):
    """Defined iterator to implement multiprocessing. Yields i in range and the
     segmented images given."""
    for i in range(N):
        yield i, foci_labeled, cell_segm, mito_segm

        
def evaluate_distance(foci_labeled, mito_segm):
    # calculate pixel superposition
    exp_pix_sup = dk.calculate_superposition(foci_labeled, mito_segm)
    exp_foc_sup = dk.calculate_superposition(foci_labeled, mito_segm, 'label')
    
    distances = dk.calculate_distances(foci_labeled, mito_segm)
    
    distances['exp_pix_sup'] = exp_pix_sup
    distances['exp_foc_sup'] = exp_foc_sup
    
    return distances


def evaluate_superposition(foci_labeled, cell_segm, mito_segm, N=500, path=None, max_dock_distance=3):
    """Pipeline that receives foci and mitocondrial stacks, segments foci,
    citoplasm and mitochondria. If path is given, segmentation is saved there.
    Superposition is evaluated and randomization of foci position is performed
    to evaluate correspondence with random positioning distribution. A
    DataFrame with calculated superpositions is returned."""

    mito_segm = np.asarray([fa.binary_dilation(this, fa.disk(max_dock_distance)) for this in mito_segm])  # Dilate
    # mitochondria to see if foci are close but not superposed

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
        fa.save_img(save_cum_dir, cum_sim, axes='ZYX', create_dir=True)

    return res


def count_foci(foci_labeled, foci_stack=None):
    """Pipeline that receives foci and mitocondrial stacks, segments foci,
    citoplasm and mitochondria. If path is given, segmentation is saved there.
    A DataFrame with foci found and their characterizations is returned."""
    if foci_stack is not None:
        df = fa.label_to_df(foci_labeled, cols=['label', 'centroid', 'coords',
                                                'area', 'mean_intensity'],
                            intensity_image=foci_stack)
    else:
        df = fa.label_to_df(foci_labeled, cols=['label', 'centroid', 'coords',
                                                'area'])

    return df


def track_and_dock(foci_stack, mito_stack, dist_dock, scales, path=None, axes='YX'):
    foci_labeled = np.zeros_like(foci_stack, dtype=int)
    cell_segm = np.zeros_like(foci_stack, dtype=bool)
    mito_segm = np.zeros_like(mito_stack, dtype=bool)
    for t, (this_foci_stack, this_mito_stack) in enumerate(zip(foci_stack, mito_stack)):
        foci_labeled[t], cell_segm[t], mito_segm[t] = fa.segment_all(this_foci_stack, this_mito_stack, subcellular=True,
                                                                     mito_filter_size=0,
                                                                     mito_opening_disk=1,
                                                                     mito_closing_disk=1)
    try:
        tracked = tk.track(foci_labeled, max_dist=2, gap=1, extra_attrs=['area', 'mean_intensity'],
                           intensity_image=mito_segm, scale=scales, subtr_drift=False)
    except:
        return pd.DataFrame()
    particle_labeled = tk.relabel_by_track(foci_labeled, tracked)

    if path:
        fa.save_all(particle_labeled, cell_segm, mito_segm, path, axes=axes)

    # Save image classifying docked foci
    tracked = tk.add_distances(tracked, particle_labeled, mito_segm)
    dock_vid = tk.relabel_video_by_dock(foci_labeled, tracked, dist_dock)
    save_dock_img_dir = path.with_name(path.stem + '_dock.tif')
    fa.save_img(save_dock_img_dir, dock_vid, axes='TZYX')

    tracked = tk.add_distances(tracked, particle_labeled, cell_segm, col_name='full_erode')

    # correct for image calibration
    tracked['distance'] = tracked['distance'].values * scales['X']
    tracked['time'] = tracked['frame'].values * scales['T']

    return tracked
