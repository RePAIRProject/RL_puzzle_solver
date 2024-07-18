
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
import cv2

from puzzle_utils.shape_utils import place_on_canvas
from puzzle_utils.pieces_utils import crop_to_content

def compute_cost_using_motifs_compatibility(idx1, idx2, pieces, mask_ij, cmp_parameters, ppars, yolov8_obb_detector, verbosity=1):

    (p, z_id, m, rot, computation_parameters) = cmp_parameters ### ERROR ???
    #(p, z_id, m, rot) = cmp_parameters

    if verbosity > 1:
        print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")
    if idx1 == idx2:
        # print('idx == ')
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1
    else:
        print(f"computing cost matrix for piece {idx1} vs piece {idx2}")
        candidate_values = np.sum(mask_ij > 0)
        image1 = pieces[idx1]['img']
        image2 = pieces[idx2]['img']
        poly1 = pieces[idx1]['polygon']
        poly2 = pieces[idx2]['polygon']

        R_cost_conf, R_cost_overlap  = motif_compatibility_measure_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, yolov8_obb_detector, verbosity=1)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")

        R_cost = R_cost_overlap

    return R_cost

#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def motif_compatibility_measure_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, yolov8_obb_detector, detect_on_crop=True, verbosity=1):
    # Get the yolo model

    R_cost_conf = np.zeros((m.shape[1], m.shape[1], len(rot)))
    R_cost_overlap = np.zeros((m.shape[1], m.shape[1], len(rot)))

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
                    #plt.imshow(pieces_ij_on_canvas)

                    
                    if detect_on_crop == True:
                        cropped_img, x0, x1, y0, y1  = crop_to_content(pieces_ij_on_canvas, return_vals=True)
                        img_pil = Image.fromarray(np.uint8(cropped_img))
                    else:
                        img_pil = Image.fromarray(np.uint8(pieces_ij_on_canvas))

                    obbs = yolov8_obb_detector(img_pil)[0]
                    

                    ### Check Poly-motif-bb intersection
                    plt.imshow(pieces_ij_on_canvas)
                    plt.plot(*piece_i_on_canvas['polygon'].boundary.xy)
                    plt.plot(*piece_j_on_canvas['polygon'].boundary.xy)
                    score_sum_conf = 0; cont1 = 0
                    score_sum_overlap = 0; cont2 = 0
                    for det_obb in obbs.obb:
                        do_pts = det_obb.cpu().xyxyxyxy.numpy()[0]
                        if detect_on_crop == True:
                            obb_shapely_points = [(point[0]+x0, point[1]+y0) for point in do_pts]
                        else:
                            obb_shapely_points = [(point[0], point[1]) for point in do_pts]
                        obb_poly = shapely.Polygon(obb_shapely_points)
                        plt.plot(*obb_poly.boundary.xy)
                        

                        inters_poly_i = shapely.is_empty(shapely.intersection(obb_poly, piece_i_on_canvas['polygon'].boundary))
                        inters_poly_j = shapely.is_empty(shapely.intersection(obb_poly, piece_j_on_canvas['polygon'].boundary))

                        if (inters_poly_j == False) and (inters_poly_i == False):
                            bb_score = det_obb.conf.item()
                            print(bb_score)
                            score_sum_conf = score_sum_conf + bb_score
                            cont1 = 1 + cont1

                        # Polygon corner points coordinates
                        pts = np.array(do_pts, dtype='int64')
                        color = (255, 255, 255)
                        im0 = np.zeros(np.shape(img_pil)[0:2], dtype='uint8')
                        im_ij_obb_mask = cv2.fillPoly(im0, [pts], color)
                        class_label = int(det_obb.cpu().cls.numpy()[0])

                        im_i_obb_mask = piece_i_on_canvas['motif_mask'][:, :, class_label]
                        im_j_obb_mask = piece_j_on_canvas['motif_mask'][:, :, class_label]
                        im_ij_obb_mask = np.clip(im_ij_obb_mask, 0, 1)
                        im_i_obb_mask = np.clip(im_i_obb_mask, 0, 1)
                        im_j_obb_mask = np.clip(im_j_obb_mask, 0, 1)
                        area_i = np.sum(im_i_obb_mask)
                        area_j = np.sum(im_j_obb_mask)
                        area_ij = np.sum(im_ij_obb_mask)
                        #
                        overlap_score = 0
                        if area_i + area_j > 0:
                            overlap_score = area_ij/(area_i + area_j)

                        print(overlap_score)
                        score_sum_overlap = score_sum_overlap + overlap_score
                        cont2 = 1 + cont2

                    plt.show()
                    breakpoint()

                    motif_conf_scores = 0
                    if cont1 > 0:
                        motif_conf_scores = score_sum_conf/cont1
                    print(motif_conf_scores)

                    motif_overlap_score = 0
                    if cont2 > 0:
                        motif_overlap_score = score_sum_overlap / cont2
                    print(motif_overlap_score)

                    #plt.show()
                    R_cost_conf[iy, ix, t] = motif_conf_scores
                    R_cost_overlap[iy, ix, t] = motif_overlap_score

    return R_cost_conf, R_cost_overlap
