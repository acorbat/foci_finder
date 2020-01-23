import pathlib
import yaml
import queue
from threading import Thread

from skimage import measure as meas

import tifffile as tif
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
    """Generates a yaml dictionary at output with all the pairs of stacks found
    at folder."""
    append_to_yaml(folder, output, {})


def append_to_yaml(folder, output, filename_or_dict):
    """Appends czi files found at folder to the yaml dictionary at
    filename_or_dict (or dictionary) and saves it at output.

    Parameters
    ----------
    folder : str
        path to folder where file are located
    output : str
        path to yaml file where dictionary is to be saved
    filename_or_dict : str or dict
        path to yaml dictionary or dictionary to which new stack paths are to be
        appended
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

        with open(output, 'w', encoding='utf-8') as fo:
            fo.write(yaml.dump(d))


def look_for_cells(img_dir, save_dir=None):
    img_file = lsm.LSM880(str(img_dir))

    file_dict = {}
    for this_scene in range(img_file.get_dim_dict()['S']):
        print('Analyzing scene %s' % this_scene)

        if save_dir:
            this_save_dir = save_dir.with_name(img_dir.stem +
                                                    '_cell_%s.tif' % this_scene)

        if save_dir and this_save_dir.exists():
            cell_labeled = tif.TiffFile(str(this_save_dir)).asarray()

        else:
            stack = img_file[{'S': this_scene, 'C': 0}]

            stack = stack.transpose('T', 'Z', 'Y', 'X').values
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
                                             for tp in range(stack.shape[0])}
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


def add_thresh_to_yaml(filename, base_dir, LoG_size=None):
    """Opens filename dictionary and adds the threshold."""
    with open(filename, 'r', encoding='utf-8') as fi:
        dinput = yaml.load(fi.read())

    dout = {}

    # Load images, process them and show for thresholding
    try:
        for ndx, (k, stack, cell_labeled) in \
                enumerate(ibuffer(2, load_stack(dinput.keys(), base_dir))):
            print('%d/%d: %s' % (ndx, len(dinput), k))

            v = dinput[k]
            for cell, tp_dict in v.items():
                cell_mask = cell_labeled == cell
                tp_dict = vs.visualizer(stack, tp_dict, cell_mask,
                                        LoG_size=LoG_size)

                v[cell] = tp_dict

            dout[k] = v

    except KeyboardInterrupt:
        pass

    with open(filename[:filename.rfind('.')] + '_threshed.yaml',
              'w', encoding='utf-8') as fo:
        fo.write(yaml.dump(dout))


def load_stack(filenames, dcrop=None):
    """Iterates over the given list of filenames and yields filename and either
    the saved crop coordinates (if dcrop has them), a normalized image to
    perform the crop or None if errors arise while getting the image."""
    for filename in filenames:
        k = filename
        if dcrop and k in dcrop and 'time_crop' in dcrop[k]:
            yield filename, dcrop[k]['time_crop']
        else:
            try:
                original = io.imread(filename)

                print('%.2f: %s' % (t.elapsed, filename))
                yield filename, original
            except:
                print('load stack did not work')
                yield filename, None
