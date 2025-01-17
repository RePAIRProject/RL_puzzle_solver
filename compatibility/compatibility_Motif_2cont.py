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


def compute_CM_using_motifs(idx1, idx2, pieces, mask_ij, ppars, yolo_obj_detector, det_type='yolo-obb', verbosity=1):
    p = ppars['p']
    z_id = ppars['z_id']
    m = ppars['m']
    rot = ppars['rot']

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

        R_cost_conf, R_cost_overlap = motifs_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1,
                                                                         idx2, yolo_obj_detector, det_type=det_type,
                                                                         verbosity=1)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")

        R_cost = R_cost_overlap

    return R_cost


#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def motifs_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, \
                                       yolo_obj_detector, det_type='yolo-obb', detect_on_crop=True, area_ratio=0.1,
                                       verbosity=1):
    # Get the yolo model

    R_cost_conf = np.zeros((m.shape[1], m.shape[1], len(rot)))
    R_cost_overlap = np.zeros((m.shape[1], m.shape[1], len(rot)))

    for t in range(len(rot)):
        theta = rot[t]  # theta_rad = theta * np.pi / 180
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
                    piece_j_on_canvas = place_on_canvas(pieces[idx2], (x_j_pixel, y_j_pixel), ppars.canvas_size,
                                                        t * ppars.theta_step)
                    overlap_area = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
                    pieces_ij_on_canvas = piece_i_on_canvas['img'] + piece_j_on_canvas['img'] * (
                        np.dstack(((overlap_area < 2), (overlap_area < 2), (overlap_area < 2)))).astype(int)
                    # pieces_ij_on_canvas *= (np.dstack(((overlap_area < 2), (overlap_area < 2), (overlap_area < 2)))).astype(int)

                    # mask_ij_on_canvas = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
                    # pieces_ij_on_canvas/= np.clip(mask_ij_on_canvas,1,2).astype(float)
                    # plt.imshow(pieces_ij_on_canvas)
                    # plt.ion()

                    if detect_on_crop == True:
                        cropped_img, x0, x1, y0, y1 = crop_to_content(pieces_ij_on_canvas, return_vals=True)
                        img_pil = Image.fromarray(np.uint8(cropped_img))
                    else:
                        x0 = 0
                        y0 = 0
                        img_pil = Image.fromarray(np.uint8(pieces_ij_on_canvas))

                    detected = yolo_obj_detector(img_pil, verbose=False)[0]

                    ### Check Poly-motif-bb intersection
                    # plt.imshow(pieces_ij_on_canvas)
                    # plt.plot(*piece_i_on_canvas['polygon'].boundary.xy)
                    # plt.plot(*piece_j_on_canvas['polygon'].boundary.xy)
                    score_sum_conf = 0;
                    cont1 = 0
                    score_sum_overlap = 0;
                    cont2 = 0
                    if det_type == 'yolo-obb':
                        det_objs = detected.obb
                    elif det_type == 'yolo-bbox':
                        det_objs = detected.boxes

                    # print(f"detected {len(det_objs)} objects")
                    for det_obb in det_objs:

                        if det_type == 'yolo-obb':
                            do_pts = det_obb.cpu().xyxyxyxy.numpy()[0]
                        elif det_type == 'yolo-bbox':
                            ps = det_obb.cpu().xyxy[0]
                            p1 = np.asarray(ps[:2])
                            p2 = np.asarray([ps[0], ps[3]])
                            p3 = np.asarray(ps[2:])
                            p4 = np.asarray([ps[2], ps[1]])
                            do_pts = np.asarray([p1, p2, p3, p4])

                        # do_pts are in cropped version!
                        # please add [x0, y0] to go back to canvas

                        if detect_on_crop == True:
                            obb_shapely_points = [(point[0] + x0, point[1] + y0) for point in do_pts]
                        else:
                            obb_shapely_points = [(point[0], point[1]) for point in do_pts]
                        det_obb_poly = shapely.Polygon(obb_shapely_points)
                        # plt.plot(*det_obb_poly.boundary.xy)

                        inters_poly_i = shapely.intersection(det_obb_poly, piece_i_on_canvas['polygon'])
                        inters_poly_j = shapely.intersection(det_obb_poly, piece_j_on_canvas['polygon'])
                        # breakpoint()

                        if (inters_poly_j.area / det_obb_poly.area > area_ratio) and \
                                (inters_poly_i.area / det_obb_poly.area > area_ratio):
                            bb_score = det_obb.conf.item()
                            # print(bb_score)
                            score_sum_conf = score_sum_conf + bb_score
                            cont1 = 1 + cont1

                            # Polygon corner points coordinates
                            # x0, y0 is from crop to canvas
                            pts = np.array(do_pts, dtype='int64') + np.array([x0, y0])
                            color = (255, 255, 255)
                            im0 = np.zeros(np.shape(pieces_ij_on_canvas)[0:2], dtype='uint8')
                            im_ij_obb_mask = cv2.fillPoly(im0, [pts], color)
                            class_label = int(det_obb.cpu().cls.numpy()[0])

                            im_i_obb_mask = piece_i_on_canvas['motif_mask'][:, :, class_label]
                            im_j_obb_mask = piece_j_on_canvas['motif_mask'][:, :, class_label]
                            im_ij_obb_mask = np.clip(im_ij_obb_mask, 0, 1)

                            sum_ij_obb_mask = np.clip(im_i_obb_mask + im_j_obb_mask, 0, 1)
                            overlap_score = 0
                            if np.sum(sum_ij_obb_mask) > 0:
                                overlap_score = np.sum(sum_ij_obb_mask * im_ij_obb_mask) / np.sum(sum_ij_obb_mask)

                            # print('sum * ', np.sum(sum_ij_obb_mask*im_ij_obb_mask))
                            # print(' /sum', np.sum(sum_ij_obb_mask))
                            # print('ovrelap score', overlap_score)
                            score_sum_overlap = score_sum_overlap + overlap_score
                            cont2 = 1 + cont2

                    motif_conf_score = 0
                    if cont1 > 0:
                        motif_conf_score = score_sum_conf / cont1
                    # print(motif_conf_score)

                    motif_overlap_score = 0
                    if cont2 > 0:
                        motif_overlap_score = score_sum_overlap / cont2
                    # print("motif overlap", motif_overlap_score)

                    # if motif_overlap_score > -1:
                    #     plt.title(f"Score: {motif_overlap_score}")
                    #     plt.show()
                    #     breakpoint()
                    #     plt.cla()
                    # else:
                    #     print(f'score 0 ({motif_overlap_score})')
                    #     plt.cla()
                    R_cost_conf[iy, ix, t] = motif_conf_score
                    R_cost_overlap[iy, ix, t] = motif_overlap_score

    return R_cost_conf, R_cost_overlap


def compute_line_based_CM_LAP(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, \
                              color1, color2, cat1, cat2, mask_ij, ppars, verbosity=1, guglielmo=3):
    compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot)))
    # for t in range(1):
    for t in range(len(rot)):
        # theta = -rot[t] * np.pi / 180      # rotation of F2
        t_rot = time.time()
        theta = rot[t]
        theta_rad = theta * np.pi / 180  # np.deg2rad(theta) ?
        for ix in range(m.shape[1]):  # (z_id.shape[0]):
            t_x = time.time()
            for iy in range(m.shape[1]):  # (z_id.shape[0]):
                t_y = time.time()
                z = z_id[iy, ix]  # ??? [iy,ix] ??? strange...
                valid_point = mask_ij[iy, ix, t]
                # print(iy, ix, t)
                if valid_point > 0:
                    # print([iy, ix, t])
                    # check if line1 crosses the polygon2
                    intersections1, useful_lines_s11, useful_lines_s12 = line_poligon_intersect(z[::-1], -theta, poly2,
                                                                                                [0, 0], 0, poly1, s11,
                                                                                                s12, ppars)

                    # return intersections
                    useful_lines_alfa1 = alfa1[intersections1]  # no rotation here!
                    useful_lines_color1 = color1[intersections1]
                    useful_lines_cat1 = cat1[intersections1]
                    useful_lines_s11 = useful_lines_s11[intersections1]
                    useful_lines_s12 = useful_lines_s12[intersections1]

                    # check if line2 crosses the polygon1
                    intersections2, useful_lines_s21, useful_lines_s22 = line_poligon_intersect([0, 0], 0, poly1,
                                                                                                z[::-1], -theta, poly2,
                                                                                                s21, s22, ppars)
                    useful_lines_alfa2 = alfa2[intersections2] + theta_rad  # the rotation!

                    useful_lines_color2 = color2[intersections2]
                    useful_lines_cat2 = cat2[intersections2]
                    useful_lines_s21 = useful_lines_s21[intersections2]
                    useful_lines_s22 = useful_lines_s22[intersections2]

                    n_lines_f1 = useful_lines_alfa1.shape[0]
                    n_lines_f2 = useful_lines_alfa2.shape[0]

                    if n_lines_f1 == 0 and n_lines_f2 == 0:
                        # tot_cost = ppars.badmatch_penalty/2 # accept with some cost, guglielmo=3
                        # tot_cost = ppars.max_dist
                        tot_cost = ppars.badmatch_penalty * 0.9

                    elif (n_lines_f1 == 0 and n_lines_f2 > 0) or (n_lines_f1 > 0 and n_lines_f2 == 0):
                        n_lines = (np.max([n_lines_f1, n_lines_f2]))
                        # tot_cost = ppars.mismatch_penalty*n_lines**2   ## it will be very high compatibility
                        tot_cost = ppars.badmatch_penalty * 0.9

                    else:
                        # Compute cost_matrix, LAP, penalty, normalize
                        dist_matrix0 = np.zeros((n_lines_f1, n_lines_f2))
                        dist_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        gamma_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        color_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        cat_matrix = np.zeros((n_lines_f1, n_lines_f2))

                        for i in range(n_lines_f1):
                            for j in range(n_lines_f2):
                                # new
                                color_matrix[i, j] = np.all(useful_lines_color1[i, :] == useful_lines_color2[j, :])
                                cat_matrix[i, j] = np.all(useful_lines_cat1[i] == useful_lines_cat2[j])
                                gamma = useful_lines_alfa1[i] - useful_lines_alfa2[j]
                                gamma_matrix[i, j] = np.abs(np.sin(gamma))

                                d1 = distance.euclidean(useful_lines_s11[i], useful_lines_s21[j])
                                d2 = distance.euclidean(useful_lines_s11[i], useful_lines_s22[j])
                                d3 = distance.euclidean(useful_lines_s12[i], useful_lines_s21[j])
                                d4 = distance.euclidean(useful_lines_s12[i], useful_lines_s22[j])

                                dist_matrix[i, j] = np.min([d1, d2, d3, d4])

                        dist_matrix[gamma_matrix > ppars.thr_coef] = ppars.badmatch_penalty
                        dist_matrix[dist_matrix > ppars.max_dist] = ppars.badmatch_penalty
                        dist_matrix[cat_matrix < 1] = ppars.badmatch_penalty  ## Check if works !!!

                        # # LAP
                        row_ind, col_ind = linear_sum_assignment(dist_matrix)
                        tot_cost = dist_matrix[row_ind, col_ind].sum()
                        # print([tot_cost])
                        # print("#" * 50)
                        # print(dist_matrix)

                        # # penalty
                        penalty = np.abs(n_lines_f1 - n_lines_f2) * ppars.mismatch_penalty  # no matches penalty
                        tot_cost = tot_cost / np.min([n_lines_f1, n_lines_f2])  # normalize to all lines in the game
                        tot_cost = (tot_cost + penalty)
                        # print(tot_cost)

                    compatibility_score = np.clip(ppars.badmatch_penalty - tot_cost, 0, ppars.badmatch_penalty)
                    compatibility_matrix[iy, ix, t] = compatibility_score
                if verbosity > 4:
                    print(f"comp on y took {(time.time() - t_y):.02f} seconds")
            if verbosity > 3:
                print(f"comp on x,y took {(time.time() - t_x):.02f} seconds")
        if verbosity > 2:
            print(
                f"comp on t = {t} (for all x,y) took {(time.time() - t_rot):.02f} seconds ({np.sum(mask_ij[:, :, t] > 0)} valid values)")

    # #print(R_cost)
    # R_cost[R_cost > ppars.badmatch_penalty] = ppars.badmatch_penalty
    # len_unique = len(np.unique(R_cost))
    # kmin_cut_val = np.sort(np.unique(R_cost))[::-1][-min(len_unique,ppars.k)]
    # norm_R_cost = np.maximum(1 - R_cost / kmin_cut_val, 0)
    # #print(norm_R_cost)

    return compatibility_matrix

