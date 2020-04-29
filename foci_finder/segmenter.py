import pathlib
import queue
from skimage import measure as meas, util
from threading import Thread
import tifffile as tif
import yaml

from img_manager import lsm880 as lsm

from . import foci_analysis as fa
from . import tracking as tk
from . import visualize as vs


def scan_folder(folder):
    """Iterator yielding every czi file in folder."""
    p = pathlib.Path(folder)
    for img_dir in p.rglob('*.czi'):
        if img_dir.is_dir() or img_dir.suffix.lower() != '.czi':
            continue

        yield img_dir


def create_base_yaml(folder, output):
    """Generates a yaml dictionary at output with all scenes and cells found
    at each file at folder."""
    append_to_yaml(folder, output, {})


def append_to_yaml(folder, output, filename_or_dict):
    """Appends czi files found at folder to the yaml dictionary at
    filename_or_dict (or dictionary) and saves it at output.

    Parameters
    ----------
    folder : str
        path to folder where files are located
    output : str
        path to yaml file where dictionary is to be saved
    filename_or_dict : str or dict
        path to yaml dictionary or dictionary to which new stack paths are to
        be appended
    """

    if isinstance(filename_or_dict, str):
        with open(filename_or_dict, 'r', encoding='utf-8') as fi:
            d = yaml.load(fi.read())
    else:
        d = filename_or_dict

    for img_dir in scan_folder(folder):
        print('Analyzing cells in image %s' % str(img_dir))
        rel_path = img_dir.relative_to(folder)
        file_dict = look_for_cells(img_dir, save_dir=output.joinpath(rel_path))
        d[str(rel_path)] = file_dict

        with open(output.joinpath('file_cell_dict.yaml'),
                  'w', encoding='utf-8') as fo:
            fo.write(yaml.dump(d))


def look_for_cells(img_dir, save_dir=None):
    """Looks for cells at each scene and returns a dictionary containing
    dictionaries for each scene, each cell and each timepoint.

    Parameters
    ----------
    img_dir : path
        path to image file to be analyzed
    save_dir : path, optional
        If given, labeled stacks of cells are saved there.

    Returns
    -------
    file_dict : dictionary
        Dictionary containing a dictionary for each scene, which contains a
        dictionary for each cell, and inside a dictionary for the threshold to
        be used at each timepoint.
    """
    img_file = lsm.LSM880(str(img_dir))

    file_dict = {}
    for this_scene in range(img_file.get_dim_dict()['S']):
        print('Analyzing scene %s' % this_scene)

        if save_dir:
            this_save_dir = save_dir.with_name(img_dir.stem +
                                                    '_cell_%s.tif' % this_scene)

        if save_dir and this_save_dir.exists():
            cell_labeled = tif.TiffFile(str(this_save_dir)).asarray()
            time_length = cell_labeled.shape[0]
            labels = []
            for region in meas.regionprops(cell_labeled[0]):
                labels.append(region.label)

        else:
            stack = img_file[{'S': this_scene, 'C': 0}]

            stack = stack.transpose('T', 'Z', 'Y', 'X').values
            time_length = stack.shape[0]
            cell_labeled = fa.find_cell(stack)

            labels = []
            for region in meas.regionprops(cell_labeled[0]):
                labels.append(region.label)

            if len(labels) > 1:
                tracked_df = tk.track(cell_labeled, max_dist=200,
                                      intensity_image=stack)
                cell_labeled = tk.relabel_by_track(cell_labeled, tracked_df)

            if save_dir:
                fa.save_img(this_save_dir, cell_labeled, axes='TZYX',
                            create_dir=True, dtype='uint8')

        file_dict[this_scene] = {this_cell: {tp: 0.0
                                             for tp in range(time_length)}
                                 for this_cell in labels}

    return file_dict


# Object used by _background_consumer to signal the source is exhausted
# to the main thread.
_sentinel = object()


class _background_consumer(Thread):
    """Will fill the queue with content of the source in a separate thread.

    >>> import Queue
    >>> q = Queue.Queue()
    >>> c = _background_consumer(q, range(3))
    >>> c.start()
    >>> q.get(True, 1)
    0
    >>> q.get(True, 1)
    1
    >>> q.get(True, 1)
    2
    >>> q.get(True, 1) is _sentinel
    True
    """

    def __init__(self, queue, source):
        Thread.__init__(self)

        self._queue = queue
        self._source = source

    def run(self):
        for item in self._source:
            self._queue.put(item)

        # Signal the consumer we are done.
        self._queue.put(_sentinel)


class ibuffer(object):
    """Buffers content of an iterator polling the contents of the given
    iterator in a separate thread.
    When the consumer is faster than many producers, this kind of
    concurrency and buffering makes sense.

    The size parameter is the number of elements to buffer.

    The source must be threadsafe.

    Next is a slow task:
    >>> from itertools import chain
    >>> import time
    >>> def slow_source():
    ...     for i in range(10):
    ...         time.sleep(0.1)
    ...         yield i
    ...
    >>>
    >>> t0 = time.time()
    >>> max(chain(*( slow_source() for _ in range(10) )))
    9
    >>> int(time.time() - t0)
    10

    Now with the ibuffer:
    >>> t0 = time.time()
    >>> max(chain(*( ibuffer(5, slow_source()) for _ in range(10) )))
    9
    >>> int(time.time() - t0)
    4

    60% FASTER!!!!!11
    """

    def __init__(self, size, source):
        self._queue = queue.Queue(size)

        self._poller = _background_consumer(self._queue, source)
        self._poller.daemon = True
        self._poller.start()

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get(True)
        if item is _sentinel:
            raise StopIteration()
        return item


def add_thresh_to_yaml(filename, base_dir, save_dir, LoG_size=None):
    """Opens filename dictionary and adds the threshold.

    Parameters
    ----------
    filename : path
        path to yaml dictionary to be loaded
    base_dir : pathlib.Path
        Path to stack files
    save_dir : pathlib.Path
        Path to labeled cell segmented files
    LoG_size : tuple, list, array; optional
        Size or sizes to be used for Laplacian of Gaussian filters
    """
    with open(filename, 'r', encoding='utf-8') as fi:
        dinput = yaml.load(fi.read())

    dout = dinput.copy()

    # Load images, process them and show for thresholding
    try:
        for ndx, (k, scene_dict) in enumerate(dinput.items()):
            print('%d/%d: %s' % (ndx, len(dinput), k))
            v = dinput[k]
            for scene, stack, cell_labeled in ibuffer(1,
                                                      load_stack(scene_dict,
                                                                 k,
                                                                 base_dir,
                                                                 save_dir)):
                if stack is None:
                    continue

                w = v[scene]
                for cell, tp_dict in w.items():
                    cell_mask = cell_labeled == cell
                    tp_dict = vs.visualizer(stack, tp_dict, cell_mask,
                                            LoG_size=LoG_size)

                    w[cell] = tp_dict
                    dout[k][scene][cell] = tp_dict

                    with open(
                            filename[:filename.rfind('.')] + '_threshed.yaml',
                            'w', encoding='utf-8') as fo:
                        fo.write(yaml.dump(dout))
            dout[k] = v

    except KeyboardInterrupt:
        pass

    with open(filename[:filename.rfind('.')] + '_threshed.yaml',
              'w', encoding='utf-8') as fo:
        fo.write(yaml.dump(dout))


def load_stack(scene_dict, filename, img_dir, segm_dir):
    """Iterator for each scene in the dictionary, and returns the loaded stack
    and labeled cell segmentation.

    Parameters
    ----------
    scene_dict : dictionary
        Dictionary for each scene
    filename : path
        Ending of the filepath
    img_dir : pathlib.Path
        Path to where all the images are saved
    segm_dir : pathlib.Path
        Path to where all the labeled segmentations are saved

    Returns
    -------
    Iterator for scene, stack and labeled segmentation
    """
    k = filename
    for scene in scene_dict.keys():

        stack_dir = img_dir.joinpath(k)
        stack = lsm.LSM880(str(stack_dir))[{'S': scene, 'C': 0}]
        stack = util.img_as_float(stack)
        this_segm_dir = segm_dir.joinpath(k).with_name(stack_dir.stem +
                                                       '_cell_%s.tif' % scene)
        cell_labeled = tif.TiffFile(str(this_segm_dir)).asarray()

        yield scene, stack, cell_labeled
        # except:
        #     print('load stack did not work')
        #     yield scene, None, None
