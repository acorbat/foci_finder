import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_laplace, gaussian_filter
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_dilation, disk


def my_KMeans(stack, clusters=2):
    """Applies K Means algorithm to a whole stack ignoring NaNs and returns a stack of the same shape with the
    classification"""
    vals = stack.flatten().reshape(1, -1).T
    real_inds = np.isfinite(vals)
    classif = KMeans(n_clusters=clusters).fit_predict(vals[real_inds][:, np.newaxis])
    vals[real_inds] = classif
    return vals.reshape(stack.shape)


def find_foci(stack, LoG_size=[2, 2, 2]):
    """Receives a single 3D stack of images and returns a same size labeled image with all the foci."""
    filtered = gaussian_laplace(stack, LoG_size, mode='nearest')  # Filter image with LoG (correlates with blobs)
    classif = my_KMeans(filtered)  # all pixels are handled as list
    classif = np.concatenate([np.zeros((LoG_size[0],) + stack.shape[1:]),
                              classif,
                              np.zeros((LoG_size[0],) + stack.shape[1:])]
                             )  # add zeros in case cell is close to upper or lower limit
    classif = binary_opening(classif)  # maybe it's unnecessary or a hyper parameter
    classif = classif[LoG_size[0]:-LoG_size[0]]  # Delete added image.
    labeled = label(classif)  # labelling in 3D

    return labeled


def label_to_df(labeled, cols=['label', 'centroid', 'coords']):
    """Returns a DataFrame where each row is a labeled object and each column in cols is the regionprop to be saved."""
    regions = regionprops(labeled)
    this_focus = {col: [region[col] for region in regions] for col in cols}
    return pd.DataFrame.from_dict(this_focus)


def find_cell(stack, mask):
    """Finds cytoplasm not considering pixels in mask."""
    cell_stack = stack.copy()
    cell_stack = gaussian_filter(cell_stack, [1, 2, 2])
    dil_mask = np.asarray([binary_dilation(this, selem=disk(2)) for this in mask]) # todo: this should be outside
    cell_stack[dil_mask] = np.nan

    cell_classif = my_KMeans(cell_stack)
    cell_classif[np.isnan(cell_classif)] = 0
    return cell_classif.astype(bool)
