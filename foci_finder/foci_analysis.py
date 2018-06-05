from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_laplace
from skimage.measure import label

def find_foci(stack):
    """Receives a single 3D stack of images and returns a same size lebelled image with all the foci."""
    LoG_filtered = gaussian_laplace(stack, [2, 2, 2], mode='nearest')
    LoG_classif = KMeans(n_clusters=2).fit_predict(LoG_filtered.flatten().reshape(1, -1).T)
    LoG_classif = LoG_classif.reshape(LoG_filtered.shape)
    labeled = label(LoG_classif)

    return labeled
