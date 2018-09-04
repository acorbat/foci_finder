import sys
sys.path.append('/home/jovyan/libs/img_manager/')
sys.path.append('/home/jovyan/libs/foci_finder')
sys.path.append('/home/jovyan/libs/onefilers')

import pda
import pathlib

from img_manager import oiffile as oif
from foci_finder import pipelines as pipe


######################## Analyze segmentation
data_dir = pathlib.Path('/home/jovyan/work/201808_z-stack')


def analyze_file(p, funcs):
    if p.suffix != '.oif':
        return None, None

    print(p)

    img = oif.OifFile(str(p))  # Load file

    # get foci and mito arrays
    stack = img.asarray()
    foci_stack = stack[0].astype('float')
    mito_stack = stack[1].astype('float')

    # correct for bkg and bleeding
    foci_stack -= 20
    mito_stack -= 15
    mito_stack = mito_stack - 0.436 * foci_stack

    df = pipe.evaluate_distance(foci_stack, mito_stack, path=p)
    df.to_pickle(str(p.with_name(p.stem + '_distance.pandas')))

    df = pipe.count_foci(foci_stack, mito_stack)
    df.to_pickle(str(p.with_name(p.stem + '_foci.pandas')))

    # calculate dilations necessary to find docked foci
    x_step = pipe.get_x_step(img)
    max_dils = int(0.3 / x_step)

    df = pipe.evaluate_superposition(foci_stack, mito_stack, path=p, max_dock_distance=max_dils)
    df.to_pickle(str(p.with_name(p.stem + '_superposition.pandas')))

    return df, None


funcs = [('date', pda.process_folder, None),
         ('condition', pda.process_folder, None),
         ('experiment', pda.process_folder, None),
         ('cell', None, analyze_file)]

df, edf = pda.process_folder(data_dir, funcs)

df.to_pickle(str(data_dir.with_suffix('.pandas')))
