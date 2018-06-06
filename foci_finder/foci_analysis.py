from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_laplace
from skimage.measure import label
from skimage.morphology import binary_opening

def find_foci(stack):
    """Receives a single 3D stack of images and returns a same size labeled image with all the foci."""
    filtered = gaussian_laplace(stack, [2, 2, 2], mode='nearest')  # Filter image with LoG (correlates with blobs)
    classif = KMeans(n_clusters=2).fit_predict(filtered.flatten().reshape(1, -1).T)  # all pixels are handled as list
    classif = classif.reshape(filtered.shape)  # reshaping classification
    classif = binary_opening(classif)  # maybe it's unnecessary or a hyper parameter
    labeled = label(classif)  # labelling in 3D

    return labeled
