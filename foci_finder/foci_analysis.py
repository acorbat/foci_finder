import h5py
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_laplace, gaussian_filter
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_closing, binary_dilation, remove_small_objects, disk
from skimage.feature import blob_log
from img_manager import tifffile as tif

def my_KMeans(stack, clusters=2):
    """Applies K Means algorithm to a whole stack ignoring NaNs and returns a stack of the same shape with the
    classification"""
    vals = stack.flatten().reshape(1, -1).T
    real_inds = np.isfinite(vals)
    real_vals = vals[real_inds]
    classif_finite = KMeans(n_clusters=clusters).fit_predict(real_vals[:, np.newaxis])

    # Check if maximum of intensity is highest class
    max_intensity_label = classif_finite[np.where(real_vals == real_vals.max())[0][0]]
    min_intensity_label = classif_finite[np.where(real_vals == real_vals.min())[0][0]]
    if not classif_finite.max() == max_intensity_label:
        classif_finite[classif_finite == max_intensity_label] = clusters
        classif_finite[classif_finite == min_intensity_label] = max_intensity_label
        classif_finite[classif_finite == clusters] = min_intensity_label

    classif = np.zeros(vals.shape)  # Nans are turned to zeros
    classif[real_inds] = classif_finite  # result from classification is added here
    return classif.reshape(stack.shape)


def LoG_normalized_filter(stack, LoG_size):
    """Normalizes the stack, applies a Laplacian of Gaussian filter and then thresholds it."""
    filtered = -1 * gaussian_laplace(stack / np.max(stack), LoG_size, mode='nearest')
    return filtered


def find_foci(stack, LoG_size=None, initial_threshold=0.01e-2, max_area=10000):
    """Receives a single 3D stack of images and returns a same size labeled image with all the foci."""
    dims = len(stack.shape)
    if dims <= 3:
        if LoG_size is None:
            LoG_size = [2, ] * dims

        filtered = LoG_normalized_filter(stack, LoG_size)  # Filter image with LoG (correlates with blobs)
        threshold = initial_threshold
        filtered[filtered < threshold] = np.nan
        classif = my_KMeans(filtered)  # all pixels are handled as list
        labeled = label(classif)  # Label segmented stack

        # We can check if the objects found are very big, then too many pizels where taken into account. By changing the
        # threshold, less pixels will be taken into account and we could actually automatically find the ideal threshold
        areas = []
        for region in regionprops(labeled):
            areas.append(region.area)

        while any(area > max_area for area in areas):
            threshold *= 2
            filtered[filtered < threshold] = np.nan
            labeled = label(my_KMeans(filtered))
            areas = []
            for region in regionprops(labeled):
                areas.append(region.area)

    else:
        labeled = np.asarray([find_foci(this_stack, LoG_size=LoG_size) for this_stack in stack])

    return labeled


def label_to_df(labeled, cols=['label', 'centroid', 'coords'], intensity_image=None):
    """Returns a DataFrame where each row is a labeled object and each column in cols is the regionprop to be saved."""
    regions = regionprops(labeled, intensity_image=intensity_image)
    this_focus = {col: [region[col] for region in regions] for col in cols}
    return pd.DataFrame.from_dict(this_focus)


def find_cell(stack, mask, gaussian_kernel=None):
    """Finds cytoplasm not considering pixels in mask."""
    dims = len(stack.shape)
    if dims <= 3:
        if gaussian_kernel is None:
            if dims == 3:
                gaussian_kernel = [4, 2, 2]
            else:
                gaussian_kernel = [2, ] * dims

        cell_stack = stack.copy()
        cell_stack = gaussian_filter(cell_stack, gaussian_kernel)
        if dims == 3:
            dil_mask = np.asarray(
                [binary_dilation(this, selem=disk(2)) for this in mask])  # todo: this should be outside
        else:
            dil_mask = binary_dilation(mask, selem=disk(2))  # todo: this should be outside
        cell_stack[dil_mask] = np.nan

        cell_classif = my_KMeans(cell_stack)
        cell_classif[np.isnan(cell_classif)] = 0
        cell_classif.astype(bool)

    else:
        cell_classif = np.asarray([find_cell(this_stack, this_mask, gaussian_kernel=gaussian_kernel)
                                   for this_stack, this_mask in zip(stack, mask)])

    return cell_classif


def find_mito(stack, cell_mask, foci_mask, filter_size=4, opening_disk=0, closing_disk=2):
    """Finds mitochondrias in stack in the segmented cell plus foci."""
    dims = len(stack.shape)
    if dims <= 3:
        # Create mask of foci and cell to know where to look for mitochondria
        cell_mask = np.ma.array(cell_mask)
        foci_mask = np.ma.array(foci_mask)
        mask = np.ma.array(np.ones(stack.shape), mask=(cell_mask + foci_mask))
        if dims == 3:
            mask = np.asarray([binary_dilation(this.mask, selem=disk(2)) for this in mask])
        else:
            mask = binary_dilation(mask, selem=disk(2))
        mito_cell_stack = stack.copy()
        mito_cell_stack[~mask] = np.nan

        # Actually classify pixels
        mito_classif = my_KMeans(mito_cell_stack)

        mito_classif[np.isnan(mito_classif)] = 0
        mito_classif = mito_classif.astype(bool)
        if dims == 3:
            mito_classif = np.asarray([binary_closing(this, selem=disk(closing_disk)) for this in mito_classif])
            mito_classif = np.asarray([remove_small_objects(this.astype(bool), min_size=filter_size)
                                       for this in mito_classif])
            mito_classif = np.asarray([binary_opening(this, selem=disk(opening_disk)) for this in mito_classif])
        else:
            mito_classif = binary_closing(mito_classif, selem=disk(closing_disk))
            mito_classif = remove_small_objects(mito_classif.astype(bool), min_size=filter_size)
            mito_classif = binary_opening(mito_classif, selem=disk(opening_disk))

    else:
        mito_classif = np.asarray([find_mito(this_stack, this_cell_mask, this_foci_mask,
                                             filter_size=filter_size, opening_disk=opening_disk)
                                   for this_stack, this_cell_mask, this_foci_mask in zip(stack, cell_mask, foci_mask)])

    return mito_classif


def segment_all(foci_stack, mito_stack, subcellular=False, foci_LoG_size=None,
                mito_filter_size=3, mito_opening_disk=2, mito_closing_disk=0):
    """Takes foci and mitochondrial stacks and returns their segmentations. If mito_stack is None, mito_segm is None. If
    subcellular is True then cell_segm is all ones as you should be zoomed into the citoplasm."""
    # TODO: Add a filter for foci size
    foci_labeled = find_foci(foci_stack, LoG_size=foci_LoG_size)

    if subcellular:
        cell_segm = np.ones_like(foci_stack)
    else:
        cell_segm = find_cell(foci_stack, foci_labeled > 0)

    if mito_stack is not None:
        mito_segm = find_mito(mito_stack, cell_segm, foci_labeled > 0,
                              filter_size=mito_filter_size,
                              opening_disk=mito_opening_disk,
                              closing_disk=mito_closing_disk)
    else:
        mito_segm = None

    return foci_labeled, cell_segm, mito_segm


def relabel(labeled, swap):
    """Takes a labeled mask and a list of tuples of the swapping labels. If a label is not swapped, it will be
    deleted."""
    out = np.zeros_like(labeled)

    for new, old in swap:
        out[labeled == old] = new

    return out


def save_img(path, stack, axes='YX', create_dir=False):
    """Saves stack as 8 bit integer in tif format."""
    # TODO: change parameter order
    stack = stack.astype('float32')

    # Fill array with new axis
    ndims = len(stack.shape)
    while ndims < 5:
        stack = stack[np.newaxis, :]
        ndims = len(stack.shape)

    # Add missing and correct axes order according to fiji
    new_axes = [ax for ax in 'TZXCY' if ax not in axes[:]]
    axes = ''.join(new_axes) + axes

    stack = tif.transpose_axes(stack, axes, asaxes='TZCYX')

    if create_dir and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    tif.imsave(str(path), data=stack, imagej=True)
    # with h5py.File(str(path), "w") as store_file:
        # store_file.create_dataset(name="image", data=stack, chunks=True, compression='gzip', dtype='int8')
        # for key, val in metadata.items():
        #     store_file["image"].attrs[key] = val


def save_all(foci_labeled, cell_segm, mito_segm, path, axes='YX', create_dir=False):
    """Saves every stack in path plus the corresponding suffix. If mito_segm is None, it does not save it."""
    foci_path = path.with_name(path.stem + '_foci_segm.tiff')
    save_img(foci_path, foci_labeled, axes=axes, create_dir=create_dir)
    cell_path = path.with_name(path.stem + '_cell_segm.tiff')
    save_img(cell_path, cell_segm, axes=axes, create_dir=create_dir)
    if mito_segm is not None:
        mito_path = path.with_name(path.stem + '_mito_segm.tiff')
        save_img(mito_path, mito_segm, axes=axes, create_dir=create_dir)


def blob_detection(foci_stack, min_sigma=7, max_sigma=9, num_sigma=4, threshold=40):
    """Applies skimage blob_log to the 3D stack. Considers image dimensions as TZYX."""

    ndims = len(foci_stack.shape)
    if ndims == 4:
        blobs = [blob_detection(stack) for stack in foci_stack]
    elif ndims < 4:
        foci_stack = np.concatenate([foci_stack,
                                  np.zeros((1, ) + foci_stack.shape[1:])]
                                    )  # add zeros in case cell is close to upper or lower limit
        blobs = blob_log(foci_stack, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
        blobs = [blob for blob in blobs if blob[0] < len(foci_stack)-1]
    else:
        raise ValueError('Too many dimensions in stack.')

    return blobs


def remove_big_objects(stack, max_size):
    """Removes objects bigger than max_size from labeled stack image."""
    mark_for_deletion = []
    for region in regionprops(stack):
        if region.area > max_size:
            mark_for_deletion.append(region.label)

    for label in mark_for_deletion:
        stack[stack == label] = 0

    return stack


def filter_with_blobs(foci_labeled, blobs, foci_filter_size=None):
    """Checks which labeled region coincide with blobs found in blobs. If foci_filter_size is given, regions bigger than
    this are previously discarded."""

    if foci_filter_size is not None:
        foci_labeled = remove_big_objects(foci_labeled, foci_filter_size)

    inds = []
    for blob in blobs:
        inds.append(tuple([int(this_blob_ind) for this_blob_ind in blob[:-1]]))
    labels = [foci_labeled[this_inds] for this_inds in inds]

    foci_filtered = np.zeros_like(foci_labeled, dtype=int)
    for correct in labels:
        if correct == 0:
            continue
        foci_filtered[foci_labeled == correct] = int(correct)

    return foci_filtered


def generate_labeled_from_blobs(blobs, shape):
    """Generates a labeled image with given shape and using the location of blobs and their radius."""
    # TODO: Adapt for 3D
    blob_labeled = np.zeros(shape)

    for blob in blobs:
        location = blob[:-1]
        radius = blob[-1] * np.sqrt(2)
        focus = disk(radius)
        disk_location = (int(location[0] - focus.shape[0] // 2), int(location[1] - focus.shape[1] // 2))
        corners = (
        disk_location[0], disk_location[0] + focus.shape[0], disk_location[1], disk_location[1] + focus.shape[1])
        if all([corner > 0 for corner in corners]) and all([corner < shape[0] for corner in corners]):
            blob_labeled[corners[0]:corners[1], corners[2]:corners[3]] += disk

    return fa.label(blob_labeled)
