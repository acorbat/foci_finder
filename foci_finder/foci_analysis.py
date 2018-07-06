import h5py
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_laplace, gaussian_filter
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_dilation, remove_small_objects, disk
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


def find_foci(stack, LoG_size=None):
    """Receives a single 3D stack of images and returns a same size labeled image with all the foci."""
    dims = len(stack.shape)
    if dims <= 3:
        if LoG_size is None:
            LoG_size = [2, ] * dims

        filtered = -1 * gaussian_laplace(stack, LoG_size,
                                         mode='nearest')  # Filter image with LoG (correlates with blobs)
        classif = my_KMeans(filtered)  # all pixels are handled as list
        classif = np.concatenate([np.zeros((LoG_size[0],) + stack.shape[1:]),
                                  classif,
                                  np.zeros((LoG_size[0],) + stack.shape[1:])]
                                 )  # add zeros in case cell is close to upper or lower limit
        classif = binary_opening(classif)  # maybe it's unnecessary or a hyper parameter
        classif = classif[LoG_size[0]:-LoG_size[0]]  # Delete added image.

        labeled = label(classif)  # Label segmented stack

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
                gaussian_kernel = [1, 2, 2]
            else:
                gaussian_kernel = [2, ] * dims

        cell_stack = stack.copy()
        cell_stack = gaussian_filter(cell_stack, gaussian_kernel)
        dil_mask = np.asarray(
            [binary_dilation(this, selem=disk(2)) for this in mask])  # todo: this should be outside
        cell_stack[dil_mask] = np.nan

        cell_classif = my_KMeans(cell_stack)
        cell_classif[np.isnan(cell_classif)] = 0
        cell_classif.astype(bool)

    else:
        cell_classif = np.asarray([find_cell(this_stack, this_mask, gaussian_kernel=gaussian_kernel)
                                   for this_stack, this_mask in zip(stack, mask)])

    return cell_classif


def find_mito(stack, cell_mask, foci_mask):
    """Finds mitochondrias in stack in the segmented cell plus foci."""
    dims = len(stack.shape)
    if dims <= 3:
        # Create mask of foci and cell to know where to look for mitochondria
        cell_mask = np.ma.array(cell_mask)
        foci_mask = np.ma.array(foci_mask)
        mask = np.ma.array(np.ones(stack.shape), mask=(cell_mask + foci_mask))
        mask = np.asarray([binary_dilation(this.mask, selem=disk(2)) for this in mask])
        mito_cell_stack = stack.copy()
        mito_cell_stack[~mask] = np.nan

        # Actually classify pixels
        mito_classif = my_KMeans(mito_cell_stack)

        mito_classif[np.isnan(mito_classif)] = 0
        mito_classif = mito_classif.astype(bool)
        mito_classif = remove_small_objects(mito_classif.astype(bool), min_size=3)
        mito_classif = np.asarray([binary_dilation(this, selem=disk(2)) for this in mito_classif])

    else:
        mito_classif = np.asarray([find_mito(this_stack, this_cell_mask, this_foci_mask)
                                   for this_stack, this_cell_mask, this_foci_mask in zip(stack, cell_mask, foci_mask)])

    return mito_classif


def segment_all(foci_stack, mito_stack):
    """Takes foci and mitochondrial stacks and returns their segmentations. If mito_stack is None, mito_segm is None."""
    foci_labeled = find_foci(foci_stack)
    cell_segm = find_cell(foci_stack, foci_labeled > 0)
    if mito_stack is not None:
        mito_segm = find_mito(mito_stack, cell_segm, foci_labeled > 0)
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


def save_img(path, stack):
    """Saves stack as 8 bit integer in tif format."""
    stack = stack.astype('int8')
    tif.imsave(str(path), data=stack)
    # with h5py.File(str(path), "w") as store_file:
        # store_file.create_dataset(name="image", data=stack, chunks=True, compression='gzip', dtype='int8')
        # for key, val in metadata.items():
        #     store_file["image"].attrs[key] = val


def save_all(foci_labeled, cell_segm, mito_segm, path):
    """Saves every stack in path plus the corresponding suffix. If mito_segm is None, it does not save it."""
    foci_path = path.with_name(path.stem + '_foci_segm.tiff')
    save_img(foci_path, foci_labeled)
    cell_path = path.with_name(path.stem + '_cell_segm.tiff')
    save_img(cell_path, cell_segm)
    if mito_segm is not None:
        mito_path = path.with_name(path.stem + '_mito_segm.tiff')
        save_img(mito_path, mito_segm)
