
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

from puzzle_utils.shape_utils import place_on_canvas

def compute_cost_wrapper_for_Motifs_compatibility(idx1, idx2, pieces, regions_mask, cmp_parameters, ppars, yolov8_obb_detector, verbosity=1):

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
        image1 = pieces[idx1]['img']
        image2 = pieces[idx2]['img']
        poly1 = pieces[idx1]['polygon']
        poly2 = pieces[idx2]['polygon']

        R_cost = motif_compatibility_measure_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, yolov8_obb_detector, verbosity=1)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")

    return R_cost

#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def motif_compatibility_measure_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, yolov8_obb_detector, verbosity=1):
    # Get the yolo model

    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))

    for t in range(len(rot)):
        theta = rot[t]     # theta_rad = theta * np.pi / 180
        for ix in range(m.shape[1]):
            for iy in range(m.shape[1]):
                z = z_id[iy, ix]
                valid_point = mask_ij[iy, ix, t]
                if valid_point > 0:

                    canv_cnt = ppars.canvas_size // 2
                    grid = z_id + canv_cnt
                    x_j_pixel, y_j_pixel = grid[iy, ix]

                    # Place on canvas pairs of pieces given position
                    center_pos = ppars.canvas_size // 2
                    piece_i_on_canvas = place_on_canvas(pieces[idx1], (center_pos, center_pos), ppars.canvas_size, 0)
                    piece_j_on_canvas = place_on_canvas(pieces[idx2], (x_j_pixel, y_j_pixel), ppars.canvas_size, t * ppars.theta_step)
                    pieces_ij_on_canvas = piece_i_on_canvas['img'] + piece_j_on_canvas['img']
                    #mask_ij_on_canvas = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
                    #pieces_ij_on_canvas/= np.clip(mask_ij_on_canvas,1,2).astype(float)
                    plt.imshow(pieces_ij_on_canvas)

                    img_pil = Image.fromarray(np.uint8(pieces_ij_on_canvas))
                    obbs = yolov8_obb_detector(img_pil)[0]

                    ### Check Poly-motif-bb intersection
                    #plt.imshow(pieces_ij_on_canvas)
                    #plt.plot(*piece_i_on_canvas['polygon'].boundary.xy)
                    #plt.plot(*piece_j_on_canvas['polygon'].boundary.xy)
                    score_sum = 0; cont = 0
                    for det_obb in obbs.obb:
                        do_pts = det_obb.cpu().xyxyxyxy.numpy()[0]
                        obb_shapely_points = [(point[0], point[1]) for point in do_pts]
                        obb_poly = shapely.Polygon(obb_shapely_points)
                        #plt.plot(*obb_poly.boundary.xy)

                        inters_poly_i = shapely.is_empty(shapely.intersection(obb_poly, piece_i_on_canvas['polygon'].boundary))
                        inters_poly_j = shapely.is_empty(shapely.intersection(obb_poly, piece_j_on_canvas['polygon'].boundary))

                        if (inters_poly_j == False) and (inters_poly_i == False):
                            bb_score = det_obb.conf.item()
                            print(bb_score)
                            score_sum = score_sum + bb_score
                            cont = 1 + cont

                    motif_conf_scores = 0
                    if cont > 0:
                        motif_conf_scores = score_sum/cont
                    print(motif_conf_scores)
                    #plt.show()
                    R_cost[iy, ix, t] = motif_conf_scores

    return R_cost
