import numpy as np

from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_laplace, gaussian_filter
from skimage.measure import label
from skimage.morphology import binary_opening, binary_dilation, disk


def my_KMeans(stack, clusters=2):
    """Applies K Means algorithm to a whole stack ignoring NaNs and returns a stack of the same shape with the
    classification"""
    vals = stack.flatten().reshape(1, -1).T
    real_inds = np.isfinite(vals)
    classif = KMeans(n_clusters=clusters).fit_predict(vals[real_inds][:, np.newaxis])
    vals[real_inds] = classif
    return vals.reshape(stack.shape)


def find_foci(stack):
    """Receives a single 3D stack of images and returns a same size labeled image with all the foci."""
    filtered = gaussian_laplace(stack, [2, 2, 2], mode='nearest')  # Filter image with LoG (correlates with blobs)
    classif = my_KMeans(filtered)  # all pixels are handled as list
    classif = binary_opening(classif)  # maybe it's unnecessary or a hyper parameter
    labeled = label(classif)  # labelling in 3D

    # TODO: if last stack has foci, opening deletes them. Should add a zeros stack above and below.
    return labeled


def find_cell(stack, mask):
    """Finds cytoplasm not considering pixels in mask."""
    cell_stack = stack.copy()
    cell_stack = gaussian_filter(cell_stack, [1, 2, 2])
    dil_mask = np.asarray([binary_dilation(this, selem=disk(2)) for this in mask]) # todo: this should be outside
    cell_stack[dil_mask] = np.nan

    cell_classif = my_KMeans(cell_stack)
    cell_classif[np.isnan(cell_classif)] = 0
    return cell_classif.astype(bool)
