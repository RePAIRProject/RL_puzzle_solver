
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
from matplotlib import cm, patches
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

        R_cost_conf, R_cost_overlap  = motifs_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, yolo_obj_detector, det_type=det_type, verbosity=1)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")
      
        R_cost = R_cost_overlap

    return R_cost

def compute_CM_using_motifs_vis(idx1, idx2, pieces, mask_ij, ppars, yolo_obj_detector, det_type='yolo-obb', verbosity=1):

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
        R_cost_conf, R_cost_overlap = motifs_compatibility_for_irregular_vis(p, z_id, m, rot, pieces, mask_ij, ppars, \
            idx1, idx2, yolo_obj_detector, det_type=det_type, verbosity=1, \
            img1 = image1, img2 = image2, poly1 = poly1, poly2 = poly2)
        print(f"computed cost matrix for piece {idx1} vs piece {idx2}")
      
        R_cost = R_cost_overlap

    return R_cost


#### NEW
## pairwise compatibility measure between two pieces with and without rotation
def motifs_compatibility_for_irregular(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, \
        yolo_obj_detector, det_type='yolo-obb', detect_on_crop=True, area_ratio=0.1, verbosity=1):
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
                    overlap_area = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
                    pieces_ij_on_canvas = piece_i_on_canvas['img'] + piece_j_on_canvas['img'] * (np.dstack(((overlap_area < 2), (overlap_area < 2), (overlap_area < 2)))).astype(int)
                    # pieces_ij_on_canvas *= (np.dstack(((overlap_area < 2), (overlap_area < 2), (overlap_area < 2)))).astype(int)
                    
                    #mask_ij_on_canvas = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
                    #pieces_ij_on_canvas/= np.clip(mask_ij_on_canvas,1,2).astype(float)
                    #plt.imshow(pieces_ij_on_canvas)
                    # plt.ion()
                    
                    if detect_on_crop == True:
                        cropped_img, x0, x1, y0, y1  = crop_to_content(pieces_ij_on_canvas, return_vals=True)
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
                    score_sum_conf = 0; cont1 = 0
                    score_sum_overlap = 0; cont2 = 0
                    if det_type == 'yolo-obb':
                        det_objs = detected.obb
                    elif det_type == 'yolo-bbox':
                        det_objs = detected.boxes
                    
                    #print(f"detected {len(det_objs)} objects")
                    for det_obb in det_objs:

                        if det_type == 'yolo-obb':
                            do_pts = det_obb.cpu().xyxyxyxy.numpy()[0]
                        elif det_type == 'yolo-bbox':
                            ps = det_obb.cpu().xyxy[0]
                            p1 = np.asarray(ps[:2])
                            p2 = np.asarray([ps[0], ps[3]])
                            p3 = np.asarray(ps[2:])
                            p4 = np.asarray([ps[2], ps[1]])
                            do_pts = np.asarray([p1,p2,p3,p4])

                        # do_pts are in cropped version! 
                        # please add [x0, y0] to go back to canvas
                        
                        if detect_on_crop == True:
                            obb_shapely_points = [(point[0]+x0, point[1]+y0) for point in do_pts]
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

                            sum_ij_obb_mask = np.clip(im_i_obb_mask+im_j_obb_mask, 0, 1)
                            overlap_score = 0
                            if np.sum(sum_ij_obb_mask) > 0:
                                overlap_score = np.sum(sum_ij_obb_mask*im_ij_obb_mask)/np.sum(sum_ij_obb_mask)

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


def motifs_compatibility_for_irregular_vis(p, z_id, m, rot, pieces, mask_ij, ppars, idx1, idx2, \
        yolo_obj_detector, det_type='yolo-obb', detect_on_crop=True, area_ratio=0.1, verbosity=1, \
        img1 = None, img2 = None, poly1 = None, poly2 = None):
    
    R_cost_conf = np.zeros((m.shape[1], m.shape[1], len(rot)))
    R_cost_overlap = np.zeros((m.shape[1], m.shape[1], len(rot)))
    plt.ion()
    counter_num = 0
    plt.figure(figsize =(36, 18))
    target_dir = os.path.join('visualization_motifs', 'rp_g39', f"P_{idx1}_{idx2}")

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
                    overlap_area = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
                    pieces_ij_on_canvas = piece_i_on_canvas['img'] + piece_j_on_canvas['img'] * (np.dstack(((overlap_area < 2), (overlap_area < 2), (overlap_area < 2)))).astype(int)
                    # pieces_ij_on_canvas *= (np.dstack(((overlap_area < 2), (overlap_area < 2), (overlap_area < 2)))).astype(int)
                    
                    #mask_ij_on_canvas = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
                    #pieces_ij_on_canvas/= np.clip(mask_ij_on_canvas,1,2).astype(float)
                    #plt.imshow(pieces_ij_on_canvas)
                    # plt.ion()
                    
                    if detect_on_crop == True:
                        cropped_img, x0, x1, y0, y1  = crop_to_content(pieces_ij_on_canvas, return_vals=True)
                        img_pil = Image.fromarray(np.uint8(cropped_img))
                    else:
                        x0 = 0
                        y0 = 0
                        img_pil = Image.fromarray(np.uint8(pieces_ij_on_canvas))

                    # breakpoint()
                    plt.subplot(131)
                    plt.title("What a human sees:", fontsize=24)
                    plt.imshow(img_pil)
                    detected = yolo_obj_detector(img_pil, verbose=False)[0]
                    
                    ### Check Poly-motif-bb intersection
                    # plt.imshow(pieces_ij_on_canvas)
                    # plt.plot(*piece_i_on_canvas['polygon'].boundary.xy)
                    # plt.plot(*piece_j_on_canvas['polygon'].boundary.xy)
                    score_sum_conf = 0; cont1 = 0
                    score_sum_overlap = 0; cont2 = 0
                    if det_type == 'yolo-obb':
                        det_objs = detected.obb
                    elif det_type == 'yolo-bbox':
                        det_objs = detected.boxes
                    
                    ax = plt.subplot(132)
                    plt.imshow(img_pil)
                    plt.title("What the algorithm sees:", fontsize=24)
                    #draw_pl = np.zeros_like(img_pil, dtype=np.uint8)
                    #print(f"detected {len(det_objs)} objects")
                    for det_obb in det_objs:

                        if det_type == 'yolo-obb':
                            do_pts = det_obb.cpu().xyxyxyxy.numpy()[0]
                            #draw_pl = cv2.polylines(draw_pl,[do_pts],True,(0,255,255))
                            plt.scatter(do_pts[:,0], do_pts[:,1], color='green')
                            #plt.plot(do_pts[:,1], do_pts[:,0])
                            det_poly = patches.Polygon(do_pts, closed=True, fill=False, color='green', linewidth=5)
                            ax.add_patch(det_poly)
                        elif det_type == 'yolo-bbox':
                            ps = det_obb.cpu().xyxy[0]
                            p1 = np.asarray(ps[:2])
                            p2 = np.asarray([ps[0], ps[3]])
                            p3 = np.asarray(ps[2:])
                            p4 = np.asarray([ps[2], ps[1]])
                            do_pts = np.asarray([p1,p2,p3,p4])

                        # do_pts are in cropped version! 
                        # please add [x0, y0] to go back to canvas
                        
                        if detect_on_crop == True:
                            obb_shapely_points = [(point[0]+x0, point[1]+y0) for point in do_pts]
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

                            sum_ij_obb_mask = np.clip(im_i_obb_mask+im_j_obb_mask, 0, 1)
                            overlap_score = 0
                            if np.sum(sum_ij_obb_mask) > 0:
                                overlap_score = np.sum(sum_ij_obb_mask*im_ij_obb_mask)/np.sum(sum_ij_obb_mask)

                            # print('sum * ', np.sum(sum_ij_obb_mask*im_ij_obb_mask))
                            # print(' /sum', np.sum(sum_ij_obb_mask))
                            # print('ovrelap score', overlap_score)
                            score_sum_overlap = score_sum_overlap + overlap_score
                            cont2 = 1 + cont2

                    
                    # plt.imshow(draw_pl)
                    # breakpoint()
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
                    plt.subplot(133)
                    plt.title("Cost Matrix", fontsize=24)
                    plt.imshow(R_cost_conf[:,:,0], cmap='RdYlGn')
                
                    breakpoint()
                    plt.clf()
    return R_cost_conf, R_cost_overlap

