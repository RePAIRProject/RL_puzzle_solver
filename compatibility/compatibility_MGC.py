
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

def compute_cost_wrapper_for_Colors_compatibility(idx1, idx2, pieces, regions_mask, cmp_parameters, ppars, seg_len, verbosity=1):

    #(p, z_id, m, rot, line_matching_pars) = cmp_parameters
    (p, z_id, m, rot) = cmp_parameters

    if verbosity > 1:
        print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")
    if idx1 == idx2:
        # print('idx == ')
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1

    else:
        print(f"computing cost matrix for piece {idx1} vs piece {idx2}")
        mask_ij = regions_mask[:, :, :, idx2, idx1]
        candidate_values = np.sum(mask_ij > 0)
        #image1 = pieces[idx1]['img']
        #image2 = pieces[idx2]['img']
        #poly1 = pieces[idx1]['polygon']
        #poly2 = pieces[idx2]['polygon']
        poly1 = pieces[idx1]['segmented_poly']
        poly2 = pieces[idx2]['segmented_poly']
        border_colors1 = pieces[idx1]['boundary_seg']
        border_colors2 = pieces[idx2]['boundary_seg']

        R_cost = colors_compatibility_measure_for_irregular(p, z_id, m, rot, poly1, poly2, border_colors1, border_colors2,
                                                            mask_ij, ppars, idx1, idx2, seg_len, verbosity=1)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")

    return R_cost

#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def colors_compatibility_measure_for_irregular(p, z_id, m, rot, poly1, poly2, border_colors1, border_colors2, mask_ij, pars, idx1, idx2, seg_len, verbosity=1):

    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))-1
    n_borders_i = len(border_colors1)
    n_borders_j = len(border_colors2)  # border_colors2.shape[0]


    piece_i_shape = poly1
    piece_j_shape = poly2
    #piece_i_shape = poly1.tolist()  # shapely.polygons(poly_p)
    #piece_j_shape = poly2.tolist()  # shapely.polygons(poly_p)
    #piece_i_shape = piece_i_shape.segmentize(max_segment_length=30)
    #piece_j_shape = piece_j_shape.segmentize(max_segment_length=30)

    for t in range(len(rot)):
        theta = rot[t]     # theta_rad = theta * np.pi / 180
        for ix in range(m.shape[1]):
            for iy in range(m.shape[1]):
                z = z_id[iy, ix]
                valid_point = mask_ij[iy, ix, t]
                if valid_point > 0:
                    piece_i_trans = transform(piece_i_shape, lambda x: x - [pars.p_hs, pars.p_hs])  # poly i to center
                    piece_j_rotate = rotate(piece_j_shape, theta, origin=[pars.p_hs, pars.p_hs])    # rotate poly j
                    piece_j_trans = transform(piece_j_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z[::-1])  # poly j to xy

                    # test = intersection(piece_j_trans, piece_i_trans, grid_size=1) - this is matching segments
                    # print(test)

                    coord_i = np.round(np.array(list(piece_i_trans.exterior.coords))/2)*2  # all vertexes in array [x y] coordinates
                    coord_j = np.round(np.array(list(piece_j_trans.exterior.coords))/2)*2
                    comp_scores_matrix = np.zeros(shape=(n_borders_i, n_borders_j), dtype=np.float16)-1
                    for i in range(coord_i.shape[0]-1):
                        seg_i = coord_i[i:i+2, :]                       # extract coordinate of segment i from poly1
                        for j in range(coord_j.shape[0]-1):
                            seg_j = coord_j[j:j+2, :]                   # extract coordinate of segment j from poly2
                            # match segment i to segment j
                            match_seg_dir = np.all(seg_i == seg_j)
                            match_seg_inv = np.all(seg_i == seg_j[::-1])

                            if match_seg_dir:
                                print("WRONG direction!!!")  ### These are wrong cases !!! - negative compatibility
                                # border_i = border_colors1[i]['colors']
                                # border_j = border_colors2[j]['colors']
                                # comp_scores_matrix[i, j] = MCG_for_irregular(border_i, border_j)

                            elif match_seg_inv:
                                border_i = border_colors1[i]['colors']  # form down to up and to the right !!!
                                border_j = border_colors2[j]['colors']  # for up to down and to the left  !!!

                                if border_i.shape[0] != seg_len or border_j.shape[0] != seg_len:
                                    print('STOP - Wrong Length of border !!!')
                                    print(border_i.shape[0])
                                    print(border_j.shape[0])
                                    if border_i.shape[0] > seg_len or border_j.shape[0] > seg_len:
                                        border_i = border_i[0:seg_len, :, :]
                                        border_j = border_j[0:seg_len, :, :]
                                    elif border_i.shape[0] < seg_len or border_j.shape[0] < seg_len:
                                        cut = np.minimum(border_i.shape[0], border_j.shape[0])
                                        border_i = border_i[0:cut, :, :]
                                        border_j = border_j[0:cut, :, :]
                                        # plt.plot(*piece_i_trans.boundary.xy, linewidth=5, color="red")
                                        # plt.plot(*piece_j_trans.boundary.xy, linewidth=5, color="blue")
                                        # plt.axis('equal')
                                        # plt.show()
                                        # plt.pause(10)
                                        # plt.close()

                                border_i_inv = np.flip(border_i, (0, 1))
                                # border_i_inv = rotate(border_i, 180, reshape=False, mode='constant')  # option - check
                                comp_scores_matrix[i, j] = MCG_for_irregular(border_i_inv, border_j)

                    MGC_scores = np.mean(comp_scores_matrix[np.where(comp_scores_matrix >= 0)])  # sum of the scores of all matching borders (This is a Distance!!!)
                    #MGC_scores = np.nanmax(comp_scores_matrix[np.where(comp_scores_matrix >= 0)])   # TEST !!!

                    # print('score for all segments')
                    print(MGC_scores)
                    if not (np.isnan(MGC_scores)):
                        R_cost[iy, ix, t] = MGC_scores

    return R_cost

# auxiliary function to pairwise_compatibility_measure
def MCG_for_irregular(border_i, border_j):
    # "Solving Square Jigsaw Puzzle by Hierarchical Loop Constraints [K.Son, J.Hays, D.B.Cooper] (2019)"

    # dum = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    dum = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    border_i = border_i / 255
    border_j = border_j / 255


    l_i = np.squeeze(border_j[:, 0, ] - border_i[:, -1, ])
    e_i = np.squeeze(0.5 * (border_i[:, -1, ] - border_i[:, -2, ] + border_j[:, 1, ] - border_j[:, 0, ]))
    grad_i = np.squeeze(border_i[:, -1, ] - border_i[:, -2, ])
    v_i = np.linalg.inv(np.cov(np.append(grad_i, dum, axis=0), rowvar=False))
    d_lr = np.trace(np.dot(np.dot((l_i - e_i), v_i), np.transpose(l_i - e_i)))

    l_j = np.squeeze(border_i[:, -1, ] - border_j[:, 0, ])
    e_j = np.squeeze(0.5 * (border_j[:, 0, ] - border_j[:, 1, ] + border_i[:, -2, ] - border_i[:, -1, ]))
    grad_j = np.squeeze(border_j[:, 0, ] - border_j[:, 1, ])
    v_j = np.linalg.inv(np.cov(np.append(grad_j, dum, axis=0), rowvar=False))
    d_rl = np.trace(np.dot(np.dot((l_j - e_j), v_j), np.transpose(l_j - e_j)))

    # directional derivatives
    rows_index = list(range(1, border_i.shape[0]))
    cols_index = list(range(0, border_i.shape[0] - 1))
    sigma_i = border_i[cols_index, :, ] - border_i[rows_index, :, ]
    sigma_j = border_j[cols_index, :, ] - border_j[rows_index, :, ]

    l_i = np.squeeze(sigma_j[:, 0, ] - sigma_i[:, -1, ])
    e_i = np.squeeze(0.5 * (sigma_i[:, -1, ] - sigma_i[:, -2, ] + sigma_j[:, 1, ] - sigma_j[:, 0, ]))
    grad_i = np.squeeze(sigma_i[:, -1, ] - sigma_i[:, -2, ])
    v_i = np.linalg.inv(np.cov(np.append(grad_i, dum, axis=0), rowvar=False))
    dd_lr = np.trace(np.dot(np.dot((l_i - e_i), v_i), np.transpose(l_i - e_i)))

    l_j = np.squeeze(sigma_i[:, -1, ] - sigma_j[:, 0, ])
    e_j = np.squeeze(0.5 * (sigma_j[:, 0, ] - sigma_j[:, 1, ] + sigma_i[:, -2, ] - sigma_i[:, -1, ]))
    grad_j = np.squeeze(sigma_j[:, 0, ] - sigma_j[:, 1, ])
    v_j = np.linalg.inv(np.cov(np.append(grad_j, dum, axis=0), rowvar=False))
    dd_rl = np.trace(np.dot(np.dot((l_j - e_j), v_j), np.transpose(l_j - e_j)))

    return d_lr + d_rl + dd_lr + dd_rl
