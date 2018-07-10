import pandas as pd
import numpy as np
import trackpy as tp

from skimage.measure import regionprops

from foci_finder import foci_analysis as fa


def relabel_by_track(labeled_mask, track_df):
    """Relabels according to particle column in track_df every frame of labeled_mask according to track_df."""
    out = np.zeros_like(labeled_mask)
    for frame, df in track_df.groupby('frame'):
        swap = [[df.particle[i], df.label[i]] for i in df.index]

        out[frame] = fa.relabel(labeled_mask[frame], swap)

    return out


def track(labeled_stack, extra_attrs=None, intensity_image=None):
    """Takes labeled_stack of time lapse, prepares a DataFrame from the labeled images, saving centroid positions to be
    used in tracking by trackpy. extra_attrs is a list of other attributes to be saved into the tracked dataframe."""
    elements = []
    for t, stack in enumerate(labeled_stack):  # save each desired attribute of each label from each stack into a dict.
        for region in regionprops(stack, intensity_image=intensity_image[t]):
            element = {'frame': t, 'label': region.label}

            centroid = {axis: pos for axis, pos in zip(['x', 'y', 'z'], reversed(region.centroid))}
            element.update(centroid)

            if extra_attrs is not None:
                extra = {attr: region[attr] for attr in extra_attrs}
                element.update(extra)

            elements.append(element)
    elements = pd.DataFrame(elements)
    elements = tp.link_df(elements, 15)
    elements['particle'] += 1

    return elements


def msd_straight_forward(r):
    """Calculate msd straightforward from shifts as explained in
    https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft"""
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = r[:-shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()

    return msds


def autocorrFFT(x):
    """Calcualte autocorrelation as in
    https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft"""
    N = len(x)
    F = np.fft.fft(x, n=2*N)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real  # now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0, N)  # divide res(m) by (N-m)
    return res / n  # this is the autocorrelation in convention A


def msd_fft(r):
    """Calculate msd through fast fourier transform as explained in
    https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft"""
    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2*D.sum()
    S1 = np.zeros(N)
    for m in range(N):
      Q = Q - D[m - 1] - D[N - m]
      S1[m] = Q / (N - m)
    return S1-2*S2


def msd_for_df(df):
    """Takes a DataFrame and calculates 3D or 2D MSD accordingly and returns the same DataFrame with the added msd
    column."""
    if 'z' in df.columns and all(np.isfinite(df.z.values)):
        r = np.asarray([df.x.values, df.y.values, df.z.values]).T
    else:
        r = np.asarray([df.x.values, df.y.values]).T
    msd = msd_fft(r)
    df['msd'] = msd
    return df


def add_msd_grouped(df, group_list):
    """Returns msd for each frame separating according to columns listed in group_list."""
    return df.groupby(group_list).apply(msd_for_df)
