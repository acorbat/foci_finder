import numpy as np


def move_focus(focus_coord, new_position):
    """Takes first element to new position, translating the whole set of coordinates with it."""
    return focus_coord - focus_coord[0] + new_position


def randomize_foci_positions(foci_df, cell_coords):
    """Takes a foci DataFrame and randomizes the positions of foci into cell_coords."""
    random_foci = foci_df.copy()
    new_poss = np.random.choice(len(cell_coords), size=len(random_foci), replace=False)
    new_coords = [move_focus(foci_coord, cell_coords[new_pos]) for foci_coord, new_pos in
                  zip(random_foci.coords.values, new_poss)]
    random_foci['coords'] = new_coords

    return random_foci


def reconstruct_label_from_df(df, shape):
    """Takes a DataFrame and creates a labeled imaged of shape with the coordinates in df.coords"""
    rec = np.zeros(shape)
    rec = np.concatenate([rec, np.zeros((4,) + rec.shape[1:])])
    for i in df.index:
        rec[tuple(df.coords[i].T)] = df.label[i]
    return rec[:-4]
