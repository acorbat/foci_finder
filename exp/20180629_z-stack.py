import pda
import pathlib

from img_manager import oiffile as oif
from foci_finder import docking as dk

data_dir = pathlib.Path('/mnt/data/Laboratorio/uVesiculas/docking/simultaneous/')


def analyze_file(p, funcs):
    if p.suffix != '.oif':
        return None

    print(p)

    img = oif.OifFile(str(p))  # Load file

    # get foci and mito arrays
    stack = img.asarray()
    foci_stack = stack[0].astype('float')
    mito_stack = stack[1].astype('float')

    df = dk.evaluate_superposition(foci_stack, mito_stack)

    df.to_pickle(str(p.with_suffix('.pandas')))

    return df, None


funcs = [('date', pda.process_folder, None),
         ('condition', pda.process_folder, None),
         ('experiment', pda.process_folder, None),
         ('_', None, analyze_file)]

df, edf = pda.process_folder(data_dir, funcs)

df.to_pickle(str(data_dir.with_suffix('.pandas')))
