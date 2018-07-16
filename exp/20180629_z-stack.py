import sys
sys.path.append('/home/jovyan/libs/img_manager/')
sys.path.append('/home/jovyan/libs/foci_finder')
sys.path.append('/home/jovyan/libs/onefilers')

import pda
import pathlib

from img_manager import oiffile as oif
from foci_finder import pipelines as pipe


######################## Analyze segmentation
data_dir = pathlib.Path('/home/jovyan/work/docking/20180711')


def analyze_file_distance(p, funcs):
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

    df = pipe.evaluate_distance(foci_stack, mito_stack, path=p)
    df.to_pickle(str(p.with_name(p.stem + '_distance.pandas')))

    df = pipe.count_foci(foci_stack, mito_stack)
    df.to_pickle(str(p.with_name(p.stem + '_foci.pandas')))

    return df, None


funcs = [('date', pda.process_folder, None),
         ('condition', pda.process_folder, None),
         ('experiment', pda.process_folder, None),
         ('cell', None, analyze_file_distance)]

df, edf = pda.process_folder(data_dir, funcs)

df.to_pickle(str(data_dir.with_name('distances.pandas')))

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
