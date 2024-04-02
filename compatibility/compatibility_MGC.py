
import numpy as np
import matplotlib.colors
import os
import configs.folder_names as fnames
from PIL import Image
import time
import pdb

import shapely
from shapely import transform
from shapely import intersection, segmentize
from shapely.affinity import rotate
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib import cm

#dataset = 'image_00000'
#puzzle = 'maps_puzzle_patterns_10pcs_pieces_gkbvur_28_02_2024_eval'
#puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, args.puzzle)
#puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, dataset, puzzle)
#pieces_folder = os.path.join(puzzle_root_folder, f"{fnames.pieces_folder}")
#pieces_files = os.listdir(pieces_folder)
#pieces_files.sort()


def compute_cost_wrapper_for_Colors_compatibility(idx1, idx2, pieces, regions_mask, cmp_parameters, ppars, verbosity=1):

    (p, z_id, m, rot, line_matching_pars) = cmp_parameters

    if verbosity > 1:
        print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")
    if idx1 == idx2:
        # print('idx == ')
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1

    else:
        mask_ij = regions_mask[:, :, :, idx2, idx1]
        candidate_values = np.sum(mask_ij > 0)
        t1 = time.time()

        image1 = pieces[idx1]['img']
        image2 = pieces[idx2]['img']
        poly1 = pieces[idx1]['polygon']
        poly2 = pieces[idx2]['polygon']
        poly1 = pieces[idx1]['segmented_poly']
        poly2 = pieces[idx2]['segmented_poly']
        print(idx1)
        print(idx2)
        print(poly1)
        print(poly2)
        #border_colors1 = np.random.rand(8, 32, 2, 3)
        #border_colors2 = np.random.rand(8, 32, 2, 3)
        border_colors1 = pieces[idx1]['boundary_seg']
        border_colors2 = pieces[idx2]['boundary_seg']
        R_cost = colors_compatibility_measure_for_irregular(p, z_id, m, rot, poly1, poly2, border_colors1, border_colors2,
                                                            mask_ij, ppars, idx1, idx2, verbosity=1)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")
        # print('scores rotation 0')
        # print(np.round(R_cost[:, :, 0]).astype(int))
        # print('scores rotation 1')
        # print(np.round(R_cost[:,:,1]).astype(int))
        # print('scores rotation 2')
        # print(np.round(R_cost[:, :, 2]).astype(int))
        # print('scores rotation 3')
        # print(np.round(R_cost[:, :, 3]).astype(int))
        # R_cost_Normilized =

        if verbosity > 1:
            print(
                f"computed cost matrix for piece {idx1} vs piece {idx2}: took {(time.time() - t1):.02f} seconds ({candidate_values:.1f} candidate values) ")
            # print(R_cost)

    return R_cost

#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def colors_compatibility_measure_for_irregular(p, z_id, m, rot, poly1, poly2, border_colors1, border_colors2, mask_ij, pars, idx1, idx2, verbosity=1):

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
        t_rot = time.time()
        theta = rot[t]     # theta_rad = theta * np.pi / 180

        for ix in range(m.shape[1]):
            t_x = time.time()
            for iy in range(m.shape[1]):
                t_y = time.time()
                z = z_id[iy, ix]
                valid_point = mask_ij[iy, ix, t]
                if valid_point > 0:
                    ## transform polygon i to center
                    piece_i_trans = transform(piece_i_shape, lambda x: x - [pars.p_hs, pars.p_hs])
                    ## rotate and transform polygon j to
                    piece_j_rotate = rotate(piece_j_shape, theta, origin=[pars.p_hs, pars.p_hs])
                    piece_j_trans = transform(piece_j_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z[::-1])

                    test = intersection(piece_j_trans, piece_i_trans, grid_size=1)
                    print(test)

                    coord_i = np.round(np.array(list(piece_i_trans.exterior.coords))/2)*2  # all vertexes in array [x y] coordinates
                    coord_j = np.round(np.array(list(piece_j_trans.exterior.coords))/2)*2

                    comp_scores_matrix = np.zeros(shape=(n_borders_i, n_borders_j), dtype=np.float16)-1
                    for i in range(coord_i.shape[0]-1):
                        seg_i = coord_i[i:i+2, :]  ## extract coordinate of segment i from poly1
                        for j in range(coord_j.shape[0]-1):
                            seg_j = coord_j[j:j+2, :]  ## extract coordinate of segment j from poly2
                            # match segment i to segment j
                            match_seg_dir = np.all(seg_i == seg_j)
                            match_seg_inv = np.all(seg_i == seg_j[::-1])

                            if match_seg_dir:
                                #print(seg_i)
                                #print(seg_j)
                                #border_i = border_colors1[i, :, :, :]
                                #border_j = border_colors1[j, :, :, :]
                                border_i = border_colors1[i]['colors']
                                border_j = border_colors2[j]['colors']
                                mcg_score = MCG_for_irregular(border_i, border_j)
                                print("WRONG direction!!!") ### These are wrong cases !!

                                # plt.plot(*piece_i_trans.boundary.xy, linewidth=5, color="red")
                                # plt.plot(*piece_j_trans.boundary.xy, linewidth=5, color="blue")
                                # plt.axis('equal')
                                # plt.show()
                                # plt.pause(1)

                                comp_scores_matrix[i, j] = mcg_score
                                 #plt.close()

                            elif match_seg_inv:
                                #print(seg_i)
                                #print(seg_j)

                                border_i = border_colors1[i]['colors']  # form down to up and to the right !!!
                                border_j = border_colors2[j]['colors']  # for up to down and to the left  !!!

                                if border_i.shape[0] != 30 or border_j.shape[0] != 30:
                                    print('STOP !!!')
                                    print(border_i.shape[0])
                                    print(border_j.shape[0])

                                    if border_i.shape[0] > 30 or border_j.shape[0] > 30:
                                        border_i = border_i[0:30, :, :]
                                        border_j = border_j[0:30, :, :]
                                    elif border_i.shape[0] < 30 or border_j.shape[0] < 30:
                                        # plt.plot(*piece_i_trans.boundary.xy, linewidth=5, color="red")
                                        # plt.plot(*piece_j_trans.boundary.xy, linewidth=5, color="blue")
                                        # plt.axis('equal')
                                        # plt.show()
                                        # plt.pause(10)

                                        border_i = np.zeros(shape=(30, 2, 3))
                                        border_j = np.zeros(shape=(30, 2, 3))
                                        # border_i0[0:border_i.shape[0], :, :] = border_i
                                        # border_j0[0:border_j.shape[0], :, :] = border_j
                                        # border_i = []
                                        # border_j = []
                                        # border_i = border_i0
                                        pdb.set_trace()# border_j = border_j0
                                        print('STOP !!!')
                                    print(border_i.shape[0])
                                    print(border_j.shape[0])
                                    print('STOP !!!')

                                    #plt.close()

                                border_i_inv = np.flip(border_i, (0, 1))
                                mcg_score = MCG_for_irregular(border_i_inv, border_j)
                                print(mcg_score)
                                
                                comp_scores_matrix[i, j] = mcg_score
                    

                    # CORRECT ERROR OF 0 MATCHING SEGMETNS !!!!!!!
                    MGC_scores = np.mean(comp_scores_matrix[np.where(comp_scores_matrix >= 0)])  # sum of the scores of all matching borders (This is a Distance!!!)
                    print('score for all segments')
                    print(MGC_scores)
                    if not(np.isnan(MGC_scores)):
                        R_cost[iy, ix, t] = MGC_scores

                    if verbosity > 4:
                        print(f"comp on y took {(time.time() - t_y):.02f} seconds")

                if verbosity > 3:
                    print(f"comp on x,y took {(time.time() - t_x):.02f} seconds")
                if verbosity > 2:
                    print(
                        f"comp on t = {t} (for all x,y) took {(time.time() - t_rot):.02f} seconds ({np.sum(mask_ij[:, :, t] > 0)} valid values)")
    # output
    return R_cost


# auxiliary function to pairwise_compatibility_measure
def MCG_for_irregular(border_i, border_j):
    # slavish application of the formulas about the optimized version of MGC according:
    # "Solving Square Jigsaw Puzzle by Hierarchical Loop Constraints [K.Son, J.Hays, D.B.Cooper] (2019)"

    # dum = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    border_i = border_i / 255
    border_j = border_j / 255
    dum = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

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
