import sys
sys.path.append('/home/jovyan/libs/img_manager/')
sys.path.append('/home/jovyan/libs/foci_finder')
sys.path.append('/home/jovyan/libs/onefilers')

import pda
import pathlib

from img_manager import oiffile as oif
from img_manager import tifffile as tif
from foci_finder import pipelines as pipe


######################## Analyze segmentation
data_dir = pathlib.Path('/mnt/data/Laboratorio/uVesiculas/docking/subcellular/')


def analyze_file_track(p, funcs):
    if p.suffix != '.oif':
        return None, None

    print(p)

    img = oif.OifFile(str(p))  # Load file
    stack = img.asarray()

    # calculate dilations necessary to find docked foci
    scales = {'X': pipe.get_x_step(img),
              'Y': pipe.get_y_step(img),
              'T': pipe.get_t_step(img)}
    axes = pipe.get_axis(img)
    if 'Z' in axes:
        stack = tif.transpose_axes(stack, axes, asaxes='CTZYX')
        axes = 'CTZYX'
        scales['Z'] = pipe.get_z_step(img)
    max_dils = int(0.3 / scales['X'])

    # get foci and mito arrays
    foci_stack = stack[0].astype('float')
    mito_stack = stack[1].astype('float')

    # correct for bkg and bleeding
    foci_stack -= 20
    mito_stack -= 15
    mito_stack = mito_stack - 0.436 * foci_stack

    df = pipe.track_and_dock(foci_stack, mito_stack, max_dils, scales, path=p, axes=axes)
    df.to_pickle(str(p.with_name(p.stem + '_distance.pandas')))

    return df, None


funcs = [('date', pda.process_folder, None),
         ('condition', pda.process_folder, None),
         ('experiment', pda.process_folder, None),
         ('cell', None, analyze_file_track)]

df, edf = pda.process_folder(data_dir, funcs)

df.to_pickle(str(data_dir.with_name('tracking.pandas')))
