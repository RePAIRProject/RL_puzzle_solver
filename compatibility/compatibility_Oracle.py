import numpy as np
import matplotlib.colors
import os
import configs.folder_names as fnames
from PIL import Image
import time

import shapely
from shapely import transform
from shapely import intersection, segmentize
from shapely.affinity import rotate
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib import cm


def compute_oracle_compatibility(idx1, idx2, pieces, mask_ij, ppars, puzzle_root_folder, verbosity=1):
    #  i, j,
    # (p, z_id, m, rot, line_matching_pars) = cmp_parameters
    p = ppars['p']
    z_id = ppars['z_id']
    m = ppars['m']
    rot = ppars['rot']

    # 1. load GT_grid
    no_patches = len(pieces)
    gt_grid = np.zeros((no_patches, 3))
    import pandas as pd
    df = pd.read_csv(os.path.join(puzzle_root_folder, f'GT/gt_grid3.txt'))
    gt_grid[:, 0] = (df.loc[:, 'x'].values).astype(int)
    gt_grid[:, 1] = (df.loc[:, 'y'].values).astype(int)
    # gt_grid[:, 2] = (df.loc[:, 'rot'].values)


    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))
    center = mask_ij.shape[1] // 2
    i_pos = [center, center, 0]
    j_pos = gt_grid[idx2] - gt_grid[idx1] + [center, center, 0]

    [ix, iy, iz] = (gt_grid[idx2] - gt_grid[idx1] + [center, center, 0]).astype(int)
    if (ix < mask_ij.shape[0] ) and (iy < mask_ij.shape[0]):

        valid_points = mask_ij[iy-2:iy+3, ix-2:ix+3, iz]

        if np.sum(valid_points) > 0:
            R_cost[iy, ix, iz] = 1
            print(f"find connection for piece {idx1} vs piece {idx2}")

    return R_cost



