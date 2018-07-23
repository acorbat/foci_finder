import multiprocessing
import inspect
from collections import OrderedDict

import pandas as pd

from foci_finder import foci_analysis as fa
from foci_finder import docking as dk


class Pipe(object):
    """Pipe object should be defined so as to receive pairs of stacks corresponding to foci and mitochondria that need
    to be analyzed in the same way. All parameters and processes to be run should be defined and Pipe should be called
    for each pair, maybe even instanced in different interpreters to multiprocess."""

    def __init__(self, attrs=['label', 'centroid', 'coords', 'area'], funcs=None):

        self.attrs = attrs
        self.funcs = funcs
        self.df = pd.DataFrame()

        # segmenting attributes
        self.subcellur_img = False
        self.mito_filter_size = 3
        self.mito_opening_disk = 2


    def process(self, foci_stack, mito_stack, p=None):
        """Enter foci stack and mito stack to be processed. Returns a DataFrame with the results"""

        # Organize stack accordingly for processing

        # Segment the stacks inserted
        self.segment(foci_stack, mito_stack)

        # Extract attributes through label_to_df
        if self.attrs is not None:
            self.df = fa.label_to_df(self.foci_labeled, cols=self.attrs,
                                     intensity_image=self.mito_segm)

        # Perform extra functions
        for func in self.funcs:
            this_df = func(self.df, self.foci_labeled, self.cell_segm, self.mito_segm)
            self.df = self.df.merge(this_df)

        return self.df


    def segment(self, foci_stack, mito_stack):
        self.foci_labeled, self.cell_segm, self.mito_segm = fa.segment_all(foci_stack, mito_stack,  # unprocessed stacks
                                                                           subcellular=self.subcellur_img,
                                                                           mito_filter_size=self.mito_filter_size,
                                                                           mito_opening_disk=self.mito_opening_disk)


    def add_attr(self, attrs):
        if not isinstance(attrs, list):
            attrs = list(attrs)
        self.attrs.extend(attrs)


    def add_func(self, funcs):
        if not isinstance(funcs, list):
            funcs = list(funcs)
        self.funcs.extend(funcs)


class AdaptedFunction(object):
    """Adapted Functions should be able to take the same parameters between different instances so as to be able to list
     them and run them one after the other. Extra parameters should be saved some way. There should be a way to print
     them and characterize them in order to dump the analysis made in Pipe."""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self._get_func_parameters(func)

    def _get_func_parameters(self, func):
        di = OrderedDict(inspect.signature(func).parameters)
        dic = OrderedDict([(key, value.default)
                           if value.default is not inspect._empty
                           else (key, None)
                           for key, value in di.items()])
        self.vars = dic


    def execute(self):
        """Executes function with saved parameters."""
        caller = 'self.func(' + ', '.join([str(value) for key, value in self.vars.items()]) + ')'

        return eval(caller)




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
        for i, superpositions in p.imap_unordered(dk.randomize_and_calculate,
                                                  my_iterator(N, foci_labeled, cell_segm, mito_segm)):
            print('Performing iteration %d' % i)
            output[i] = superpositions

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
