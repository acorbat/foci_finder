import sys
sys.path.append('/home/jovyan/libs/img_manager/')
sys.path.append('/home/jovyan/libs/foci_finder')
sys.path.append('/home/jovyan/libs/onefilers')

import pda
import pathlib

from img_manager import fv1000 as fv
from img_manager import tifffile as tif
from foci_finder import pipelines as pipe


######################## Analyze segmentation
data_dir = pathlib.Path('/home/jovyan/work/201808_z-stack')


def analyze_file(p, funcs):
    if p.suffix != '.oif':
        return None, None

    print('processing: ' + str(p))

    # Load image and segmentations
    labeled_dir = p.parents[4]
    labeled_dir = labeled_dir.joinpath('randomization')
    parts = p.parts[-4:]
    for i in range(3):
        labeled_dir = labeled_dir.joinpath(parts[i])
    foci_labeled_dir = labeled_dir.joinpath(p.stem + '_foci_segm.tiff')
    cell_labeled_dir = labeled_dir.joinpath(p.stem + '_cell_segm.tiff')
    mito_labeled_dir = labeled_dir.joinpath(p.stem + '_mito_segm.tiff')

    save_dir = p.parents[4]
    save_dir = save_dir.joinpath('results')
    parts = p.parts[-4:]
    for i in range(3):
        save_dir = save_dir.joinpath(parts[i])
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    foci_img_labeled = tif.TiffFile(str(foci_labeled_dir))
    foci_labeled = foci_img_labeled.asarray().astype(int)
    cell_img_labeled = tif.TiffFile(str(cell_labeled_dir))
    cell_segm = cell_img_labeled.asarray().astype(int)
    mito_img_labeled = tif.TiffFile(str(mito_labeled_dir))
    mito_segm = mito_img_labeled.asarray().astype(int)

    img = fv.FV1000(str(p))  # Load file
    stack = img.transpose_axes('CZYX')

    foci_stack = stack[0]
    mito_stack = stack[1]

    df = pipe.count_foci(foci_labeled, foci_stack=foci_stack)
    df.to_pickle(str(save_dir.with_name(p.stem + '_foci.pandas')))

    # calculate dilations necessary to find docked foci
    x_step = img.get_x_step()
    max_dils = int(0.3 / x_step)

    df = pipe.evaluate_superposition(foci_labeled, cell_segm, mito_segm, path=save_dir.joinpath(p.name),
                                     max_dock_distance=max_dils)
    df.to_pickle(str(save_dir.with_name(p.stem + '_superposition.pandas')))

    return df, None


funcs = [('date', pda.process_folder, None),
         ('condition', pda.process_folder, None),
         ('experiment', pda.process_folder, None),
         ('cell', None, analyze_file)]

df, edf = pda.process_folder(data_dir, funcs)

df.to_pickle(str(data_dir.with_suffix('.pandas')))
