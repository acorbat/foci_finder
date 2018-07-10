import sys
sys.path.append('/mnt/data/Laboratorio/img_manager/')
sys.path.append('/mnt/data/Laboratorio/uVesiculas/foci_finder/foci_finder')
sys.path.append('/mnt/data/Laboratorio/onefilers')

import pda
import pathlib

from img_manager import oiffile as oif
from foci_finder import pipelines as pipe


######################## Analyze timelapses
data_dir = pathlib.Path('/mnt/data/Laboratorio/uVesiculas/docking/timelapses/')


def analyze_file_timelapse(p, funcs):
    if p.suffix != '.oif' or 'foci' not in p.stem:
        return None, None

    print(p)

    img = oif.OifFile(str(p))  # Load file

    # get foci and none mito arrays
    stack = img.asarray()
    foci_stack = stack[0].astype('float')
    mito_stack = None

    df = pipe.count_foci(foci_stack, mito_stack)
    df.to_pickle(str(p.with_name(p.stem + '_foci.pandas')))

    return df, None


funcs = [('date', pda.process_folder, None),
         ('condition', pda.process_folder, None),
         ('cell', None, analyze_file_timelapse)]

df, edf = pda.process_folder(data_dir, funcs)

df.to_pickle(str(data_dir.with_suffix('.pandas')))


######################## Analyze subcellular
data_dir = pathlib.Path('/mnt/data/Laboratorio/uVesiculas/docking/subcellular/')


def analyze_file_subcellular(p, funcs):
    if p.suffix != '.oif':
        return None, None

    print(p)

    img = oif.OifFile(str(p))  # Load file

    # get foci and mito arrays
    stack = img.asarray()
    foci_stack = stack[0].astype('float')
    mito_stack = stack[1].astype('float')

    if len(foci_stack.shape) == 4:
        foci_stack = foci_stack.swapaxes(1, 0)
        mito_stack = mito_stack.swapaxes(1, 0)

    df = pipe.track_and_dock(foci_stack, mito_stack, path=p)
    df.to_pickle(str(p.with_name(p.stem + '_track_and_dock.pandas')))

    return df, None


funcs = [('date', pda.process_folder, None),
         ('condition', pda.process_folder, None),
         ('experiment', pda.process_folder, None),
         ('cell', None, analyze_file_subcellular)]

df, edf = pda.process_folder(data_dir, funcs)

df.to_pickle(str(data_dir.with_suffix('.pandas')))


######################## Analyze simultaneous
data_dir = pathlib.Path('/mnt/data/Laboratorio/uVesiculas/docking/simultaneous/')


def analyze_file_simultaneous(p, funcs):
    if p.suffix != '.oif':
        return None, None

    print(p)

    img = oif.OifFile(str(p))  # Load file

    # get foci and mito arrays
    stack = img.asarray()
    # foci_stack = stack[0].astype('float')
    # mito_stack = None
    foci_stack = stack[0].astype('float')
    mito_stack = stack[1].astype('float')

    df = pipe.evaluate_superposition(foci_stack, mito_stack, path=p)
    df.to_pickle(str(p.with_name(p.stem + '_superposition.pandas')))

    df = pipe.count_foci(foci_stack, mito_stack)
    df.to_pickle(str(p.with_name(p.stem + '_foci.pandas')))

    return df, None


funcs = [('date', pda.process_folder, None),
         ('condition', pda.process_folder, None),
         ('experiment', pda.process_folder, None),
         ('cell', None, analyze_file_simultaneous)]

df, edf = pda.process_folder(data_dir, funcs)

df.to_pickle(str(data_dir.with_suffix('.pandas')))
