from skimage.transform import hough_line
from skimage.transform import hough_line
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib import cm
import shapely
from shapely import transform
from shapely.affinity import rotate
import pdb 
import math 
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
#from itertools import compress
import time 
from .shape_utils import place_on_canvas

class CfgParameters(dict):
    __getattr__ = dict.__getitem__

# def calc_line_matching_parameters(parameters, cmp_cost='new'):
#     lm_pars = CfgParameters()
#     lm_pars['thr_coef'] = 0.13
#     #lm_pars['max_dist'] = 0.70*parameters.xy_step ## changed *0.7
#     if (parameters.xy_step)>6:
#         lm_pars['max_dist'] = 6   ## changed *0.7*parameters.xy_step
#     else:
#         lm_pars['max_dist'] = 1.70*(parameters.xy_step)
#
#     lm_pars['badmatch_penalty'] = max(5, lm_pars['max_dist'] * 5 / 3) # parameters.piece_size / 3 #?
#     lm_pars['mismatch_penalty'] = 1      ## FOR REPAIR ONLY !!!
#     #lm_pars['mismatch_penalty'] = max(4, lm_pars['max_dist'] * 4 / 3) # parameters.piece_size / 4 #?
#     lm_pars['rmax'] = .5 * lm_pars['max_dist'] * 7 / 6
#     lm_pars['cmp_cost'] = cmp_cost
#     lm_pars['k'] = 3
#     return lm_pars

def create_lines_only_image(img, lines):

    black = np.zeros_like(img) 
    only_line_image = np.zeros_like(img)
    for line in lines:
        p1 = line[:2] 
        p2 = line[2:4]
        black = cv2.line(black, np.asarray([p1[0], p1[1]]).astype(int), np.asarray([p2[0], p2[1]]).astype(int), color=(1, 1, 1), thickness=1)
    only_line_image = img * black
    only_line_image += (255 - black)
    # only_line_image = 255 - only_line_image
    return only_line_image                   

def draw_lines(lines_dict, img_shape, thickness=1, color=255, use_color=False):
    angles, dists, p1s, p2s, colors, cats = extract_from(lines_dict)
    if use_color == True:
        print("WARNING: probably not working! Check the image creation")
        lines_img = np.zeros(shape=img_shape, dtype=np.uint8)
        if len(colors) > 0:
            j = 0
            assert(len(colors)==len(p1s)), f"different numbers of colors ({len(colors)}) and lines ({len(p1s)}) in the .json file!"
    lines_img = np.zeros(shape=img_shape[:2], dtype=np.uint8)
    for p1, p2 in zip(p1s, p2s):
        if use_color == False:
            lines_img = cv2.line(lines_img, np.round(p1).astype(int), np.round(p2).astype(int), color=(1), thickness=thickness)        
        else:
            if len(colors) > 0:
                color = colors[j]
                j += 1
            lines_img = cv2.line(lines_img, np.round(p1).astype(int), np.round(p2).astype(int), color=(color), thickness=thickness)        
    #cv2.imwrite(os.path.join(lin_output, f"{pieces_names[k][:-4]}_l.jpg"), 255-lines_img)
    return lines_img 

def extract_from(lines_dict):
    """
    It just unravels the different parts of the extracted line dictionary 
    """
    angles = np.asarray(lines_dict['angles'])
    dists = np.asarray(lines_dict['dists'])
    p1s = np.asarray(lines_dict['p1s'])
    p2s = np.asarray(lines_dict['p2s'])
    # optional but used almost always now
    if 'categories' in lines_dict.keys():
        cats = np.asarray(lines_dict['categories'])
    else:
        print("Warning, missing categories")
        cats = []
    if 'colors' in lines_dict.keys():
        colors = np.asarray(lines_dict['colors'])
    else:
        print("Warning, empty colors in the lines!")
        colors = []
    
    return angles, dists, p1s, p2s, colors, cats


"""
THIS IS JUST IF WE NEED TO DEBUG IT!
###
def line_poligon_intersect(z_p, theta_p, poly_p, z_l, theta_l, s1, s2, pars, poly_l):
    # check if line crosses the polygon
    # z_p1 = [0,0],  z_l2 = z,
    # z_p2 = z,   z_l1 = [0,0],
    intersections = []
    useful_lines_s1 = []
    useful_lines_s2 = []
    piece_j_shape = poly_p.tolist() #shapely.polygons(poly_p)
    piece_j_rotate = rotate(piece_j_shape, theta_p, origin=[pars.p_hs, pars.p_hs])
    piece_j_trans = transform(piece_j_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_p)
    debug = False
    if theta_p != 0 and debug == True:
        plt.plot(*(piece_j_shape.boundary.xy), linewidth=7, color='orange')
        plt.plot(*(piece_j_trans.boundary.xy), linewidth=5, color='red')
        
        poly_lines = poly_l.tline_poligon_intersectolist()
        poly_lines_rot = rotate(poly_lines, theta_l, origin=[pars.p_hs, pars.p_hs])
        poly_lines_tra = transform(poly_lines_rot, lambda x: x - [pars.p_hs, pars.p_hs] + z_l)

        plt.plot(*(poly_lines_tra.boundary.xy), linewidth=7, color='green')
        plt.plot(*(poly_lines.boundary.xy), linewidth=7, color='orange')

    for (p1, p2) in zip(s1, s2):

        # p1 = [candidate_xs[0], candidate_ys[0]]
        # p2 = [candidate_xs[1], candidate_ys[1]]
        # candidate_line_shapely0 = shapely.LineString((candidate_xy_start, candidate_xy_end))
        candidate_line_shapely0 = shapely.LineString((p1, p2))
        candidate_line_rotate = rotate(candidate_line_shapely0, theta_l, origin=[pars.p_hs, pars.p_hs])
        candidate_line_trans = transform(candidate_line_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_l)

        if theta_p != 0 and debug == True:
            plt.plot(*(candidate_line_shapely0.xy), linewidth=7, color='orange')
            plt.plot(*(candidate_line_trans.xy), linewidth=7, color='blue')
        # append to the useful lines
        useful_lines_s1.append(np.array(candidate_line_trans.coords)[0])
        useful_lines_s2.append(np.array(candidate_line_trans.coords)[-1])

        
        # OLD VERSION: 
        # if shapely.is_empty(shapely.intersection(candidate_line_trans, piece_j_trans.buffer(pars.border_tolerance))):
        # issue: if the line is completely within the polygon, it returns True (as the geometry intersection exists)
        # NEW VERSION
        # we do intersection with the `boundary` of the polygon (is it faster? at least it could be, plus can remove false positive, if any)
        if shapely.is_empty(shapely.intersection(candidate_line_trans, piece_j_trans.boundary.buffer(pars.border_tolerance))):
            intersections.append(False)
        else:
            intersections.append(True)
            if theta_p != 0 and debug == True:
                plt.plot(*(candidate_line_trans.xy), linewidth=3, color='yellow')

    if theta_p != 0 and debug == True:
        plt.axis('equal')
        plt.title(f"polygon rot: {theta_p}, line_rot: {theta_l}")
        plt.show()
        pdb.set_trace()
    return intersections, np.array(useful_lines_s1), np.array(useful_lines_s2)
"""

def line_poligon_intersect(z_p, theta_p, poly_p, z_l, theta_l, poly_l, s1, s2, pars, extrapolate=True, \
                            return_shapes=False):
    # check if line crosses the polygon
    # z_p1 = [0,0],  z_l2 = z,
    # z_p2 = z,   z_l1 = [0,0],
    intersections = []
    useful_lines_s1 = []
    useful_lines_s2 = []
    piece_j_shape = poly_p #.tolist() #shapely.polygons(poly_p)
    piece_j_rotate = rotate(piece_j_shape, theta_p, origin=[pars.p_hs, pars.p_hs])
    piece_j_trans = transform(piece_j_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_p)

    if return_shapes == True:
        trans_lines = []
        trans_useful_lines = []

    for (p1, p2) in zip(s1, s2):
        
        candidate_line_shapely0 = shapely.LineString((p1, p2))
        candidate_line_rotate = rotate(candidate_line_shapely0, theta_l, origin=[pars.p_hs, pars.p_hs])
        candidate_line_trans = transform(candidate_line_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_l)

        # append to the useful lines
        ps1 = np.array(candidate_line_trans.coords)[0]
        ps2 = np.array(candidate_line_trans.coords)[-1]
        useful_lines_s1.append(ps1)
        useful_lines_s2.append(ps2)
        # print(f"Before: p1: {p1}, p2: {p2}")
        # print(f"After: p1: {ps1}, p2: {ps2}")
        # print(f"Transf: - {pars.p_hs} + {z_l}")
        # pdb.set_trace()
        if np.isclose(distance.euclidean(ps1, ps2), 0):
            intersections.append(False)
            # print("point/line")
        else:
            dist_centers = distance.euclidean(z_p,z_l)
            candidate_poly_l_shapely0 = poly_l #.tolist()
            candidate_poly_l_rotate = rotate(candidate_poly_l_shapely0, theta_l, origin=[pars.p_hs, pars.p_hs])
            candidate_poly_l_trans = transform(candidate_poly_l_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_l)

            if extrapolate == True:
                candidate_line_extrap = getExtrapoledLine(candidate_line_trans, dist_centers, candidate_poly_l_trans, pars.border_tolerance)
            else:
                candidate_line_extrap = candidate_line_trans

            if shapely.is_empty(shapely.intersection(candidate_line_extrap, piece_j_trans.boundary)):
                intersections.append(False)
                if return_shapes == True:
                    trans_lines.append(candidate_line_extrap)
            else:
                intersections.append(True)
                if return_shapes == True:
                    trans_useful_lines.append(candidate_line_extrap)

    if return_shapes == True:
        return intersections, np.array(useful_lines_s1), np.array(useful_lines_s2), piece_j_trans, trans_lines, trans_useful_lines
    return intersections, np.array(useful_lines_s1), np.array(useful_lines_s2)

def line_poligon_intersect_vis(z_p, theta_p, poly_p, z_l, theta_l, poly_l, s1, s2, pars, extrapolate=True, \
                            return_shapes=False, draw_lines=False, draw_polygon=False, drawing_col='blue'):
    # check if line crosses the polygon
    # z_p1 = [0,0],  z_l2 = z,
    # z_p2 = z,   z_l1 = [0,0],
    # plt.ion()
    intersections = []
    useful_lines_s1 = []
    useful_lines_s2 = []
    piece_j_shape = poly_p #.tolist() #shapely.polygons(poly_p)
    piece_j_rotate = rotate(piece_j_shape, theta_p, origin=[pars.p_hs, pars.p_hs])
    piece_j_trans = transform(piece_j_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_p)
    
    if return_shapes == True:
        trans_lines = []
        trans_useful_lines = []

    # plt.subplot(121)
    # plt.title("Original")
    # plt.plot(*piece_j_shape.boundary.xy)
    # plt.subplot(122) 
    # plt.title(f"Transformation (z_p: {z_p}, z_l: {z_l}, theta_p: {theta_p})")   
    # plt.plot(*piece_j_trans.boundary.xy)
    for (p1, p2) in zip(s1, s2):
        
        candidate_line_shapely0 = shapely.LineString((p1, p2))
        candidate_line_rotate = rotate(candidate_line_shapely0, theta_l, origin=[pars.p_hs, pars.p_hs])
        candidate_line_trans = transform(candidate_line_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_l)

        # plt.subplot(121)
        # plt.plot(*candidate_line_shapely0.xy)
        # append to the useful lines
        
        ps1 = np.array(candidate_line_trans.coords)[0]
        ps2 = np.array(candidate_line_trans.coords)[-1]
        useful_lines_s1.append(ps1)
        useful_lines_s2.append(ps2)
        # print(f"Before: p1: {p1}, p2: {p2}")
        # print(f"After: p1: {ps1}, p2: {ps2}")
        # print(f"Transf: - {pars.p_hs} + {z_l}")
        # pdb.set_trace()
        if np.isclose(distance.euclidean(ps1, ps2), 0):
            intersections.append(False)
            # print("point/line")
        else:
            dist_centers = distance.euclidean(z_p,z_l)
            candidate_poly_l_shapely0 = poly_l #.tolist()
            candidate_poly_l_rotate = rotate(candidate_poly_l_shapely0, theta_l, origin=[pars.p_hs, pars.p_hs])
            candidate_poly_l_trans = transform(candidate_poly_l_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_l)
            
            if draw_polygon == True:
                plt.plot(*candidate_poly_l_trans.boundary.xy, linewidth=7, color=drawing_col)
            
            if extrapolate == True:
                candidate_line_extrap = getExtrapoledLine(candidate_line_trans, dist_centers, candidate_poly_l_trans, pars.border_tolerance)
            else:
                candidate_line_extrap = candidate_line_trans

            if draw_lines == True:
                plt.plot(*candidate_line_extrap.xy, linewidth='3', color=drawing_col)

            # breakpoint()
            if shapely.is_empty(shapely.intersection(candidate_line_extrap, piece_j_trans.boundary)):
                intersections.append(False)
                if return_shapes == True:
                    trans_lines.append(candidate_line_extrap)
                # canvas_aligned_line = []
                # for p in candidate_line_extrap.xy: 
                #     canvas_aligned_line.append(np.asarray(p)+pars.canvas_size // 2)
                    
            else:
                intersections.append(True)
                if return_shapes == True:
                    trans_useful_lines.append(candidate_line_extrap)
                # canvas_aligned_line = []
                # for p in candidate_line_extrap.xy: 
                #     canvas_aligned_line.append(np.asarray(p)+pars.canvas_size // 2)
                # if draw_lines == True:
                #     plt.plot(canvas_aligned_line, canvas_aligned_line, linewidth='5', color=drawing_col)
                # plt.subplot(122)
                # plt.plot(*candidate_line_trans.xy, linewidth=5, color="red")
                # plt.plot(*candidate_line_extrap.xy, linewidth=2, color="blue")
                # print('drawn')
    # plt.plot(*candidate_poly_l_trans.boundary.xy)
    # plt.plot(*candidate_line_trans.xy, linewidth=5, color="red")
    #breakpoint()
    

    #plt.plot(*candidate_line_extrap.xy, linewidth=2, color="blue")
    # plt.axis('equal')
    # #plt.show()
    # breakpoint()
    # plt.cla()
    plt.axis('equal')
    plt.xlim([-300, 300])
    plt.ylim([-300, 300])
    if return_shapes == True:
        return intersections, np.array(useful_lines_s1), np.array(useful_lines_s2), piece_j_trans, trans_lines, trans_useful_lines
    return intersections, np.array(useful_lines_s1), np.array(useful_lines_s2)

def getExtrapoledLine(line, dist, poly, border_tolerance):

    'Creates a line extrapoled'
    p1 = line.coords[0]
    p2 = line.coords[1]

    line_importance = distance.euclidean(p1, p2)
    if np.isclose(line_importance, 0):
        pdb.set_trace()
    dist_ratio = dist / line_importance / 4
    if line_importance < border_tolerance*2:
        dist_ratio*=0.1

    a = p2
    b = p1

    # if p1 touches the boundary, we move it away from p2!
    if not shapely.is_empty(shapely.intersection(shapely.Point(p1), poly.boundary.buffer(border_tolerance))):
        b = (p1[0]+dist_ratio*(p1[0]-p2[0]), p1[1]+dist_ratio*(p1[1]-p2[1]))
    
    # if p2 also touches the boundary, we move it away from p1!
    if not shapely.is_empty(shapely.intersection(shapely.Point(p2), poly.boundary.buffer(border_tolerance))):
        a = (p2[0]+dist_ratio*(p2[0]-p1[0]), p2[1]+dist_ratio*(p2[1]-p1[1]))

    # # p2 is the final point, move away from p1
    # if not shapely.is_empty(shapely.intersection(shapely.Point(p1), poly.boundary.buffer(border_tolerance))):
    #     a = (p2[0]+dist_ratio*(p1[0]-p2[0]), p2[1]+dist_ratio*(p1[1]-p2[1]))
    # else:
    #     a = p1

    # if not shapely.is_empty(shapely.intersection(shapely.Point(p2), poly.boundary.buffer(border_tolerance))):
    #     b = (p1[0]+dist_ratio*(p2[0]-p1[0]), p1[1]+dist_ratio*(p2[1]-p1[1]))
    # else:
    #     b = p2

    return shapely.LineString([a, b])


def compute_cost_matrix_LAP_vis(grid_xy, rot, lines_pi, lines_pj, piece_i, piece_j, mask_ij, ppars, verbosity=1):

    alfa1, r1, s11, s12, color1, cat1 = lines_pi
    alfa2, r2, s21, s22, color2, cat2 = lines_pj
    R_cost = np.ones((grid_xy.shape[1], grid_xy.shape[1], len(rot))) * (ppars.badmatch_penalty + 1)
    plt.ion()
    for t in range(len(rot)):
        theta = rot[t]
        theta_rad = rot[t] * np.pi / 180     # np.deg2rad(theta) ?
        for ix in range(grid_xy.shape[1]): 
            for iy in range(grid_xy.shape[1]):
                xy = grid_xy[iy, ix]            # ??? [iy,ix] ??? strange...
                valid_point = mask_ij[iy, ix, t]
                if valid_point > 0:

                    center_pos = ppars.canvas_size // 2
                    piece_i_on_canvas = place_on_canvas(piece_i, (center_pos, center_pos), ppars.canvas_size, 0)
                    piece_j_on_canvas = place_on_canvas(piece_j, (center_pos + xy[0], center_pos + xy[1]), ppars.canvas_size, theta)
                    overlap_area = piece_i_on_canvas['mask'] + piece_j_on_canvas['mask']
                    pieces_ij_on_canvas = piece_i_on_canvas['img'] + piece_j_on_canvas['img'] * (np.dstack(((overlap_area < 2), (overlap_area < 2), (overlap_area < 2)))).astype(int)
                    # plt.imshow(pieces_ij_on_canvas)
                    # plt.plot(*piece_i_on_canvas['polygon'].boundary.xy)
                    # plt.plot(*piece_j_on_canvas['polygon'].boundary.xy)

                    intersections1, useful_lines_s11, useful_lines_s12 = \
                        line_poligon_intersect_vis(xy[::-1], -theta, piece_j['polygon'], [0, 0],  0, \
                            piece_i['polygon'], s11, s12, ppars, extrapolate=True, draw_lines=True, \
                                draw_polygon=True, drawing_col='blue')
                    
                    # return intersections                    
                    useful_lines_alfa1 = alfa1[intersections1]  # no rotation here!
                    useful_lines_color1 = color1[intersections1]
                    useful_lines_cat1 = cat1[intersections1]
                    useful_lines_s11 = useful_lines_s11[intersections1]
                    useful_lines_s12 = useful_lines_s12[intersections1]

                    # check if line2 crosses the polygon1
                    intersections2, useful_lines_s21, useful_lines_s22 = \
                        line_poligon_intersect_vis([0, 0], 0, piece_i['polygon'], xy[::-1], -theta, \
                            piece_j['polygon'], s21, s22, ppars, extrapolate=True, draw_lines=True, \
                                draw_polygon=True, drawing_col='orange')

                    breakpoint()

                    useful_lines_alfa2 = alfa2[intersections2] + theta_rad # the rotation!
                    useful_lines_color2 = color2[intersections2]
                    useful_lines_cat2 = cat2[intersections2]
                    useful_lines_s21 = useful_lines_s21[intersections2]
                    useful_lines_s22 = useful_lines_s22[intersections2]

                    n_lines_f1 = useful_lines_alfa1.shape[0]
                    n_lines_f2 = useful_lines_alfa2.shape[0]

                    if n_lines_f1 == 0 and n_lines_f2 == 0:
                        #tot_cost = ppars.max_dist * 2  
                        tot_cost = ppars.badmatch_penalty / 3                   # accept with some cost
                    
                    elif (n_lines_f1 == 0 and n_lines_f2 > 0) or (n_lines_f1 > 0 and n_lines_f2 == 0):
                        n_lines = (np.max([n_lines_f1, n_lines_f2]))
                        tot_cost = ppars.mismatch_penalty * n_lines

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
                        #print([tot_cost])
                        #print("#" * 50)
                        #print(dist_matrix)
                        
                        # # penalty
                        penalty = np.abs(n_lines_f1 - n_lines_f2) * ppars.mismatch_penalty  # no matches penalty
                        tot_cost = (tot_cost + penalty)
                        tot_cost = tot_cost / np.max([n_lines_f1, n_lines_f2])  # normalize to all lines in the game
                        #print(tot_cost)

                    if n_lines_f1 > 0 and n_lines_f2 > 0:
                        cost_string = f"R[{iy},{ix},{t}] = {tot_cost} (penalty was {penalty}, n1={n_lines_f1}, n2={n_lines_f2})"
                        #f"Cost: {tot_cost} = {dist_matrix[row_ind, col_ind].sum()} + {penalty} ({np.abs(n_lines_f1 - n_lines_f2)})"
                    else:
                        cost_string = f"R[{iy},{ix},{t}] = {tot_cost} (penalty was {penalty}, n1={n_lines_f1}, n2={n_lines_f2})"
                        # f"Cost: {tot_cost} ({n_lines_f1} lines in pi and {n_lines_f2} lines in pj)"
                    R_cost[iy,ix,t] = tot_cost
                    plt.title(cost_string)
                    print(cost_string)
                    breakpoint()
                    plt.cla()

    plt.title("R_cost")
    plt.imshow(R_cost)
    breakpoint()

def compute_cost_matrix_LAP_debug(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, mask_ij, pars, verbosity=1, show=False):
    # ppars is the old cfg (with the parameters)
    R_cost = np.ones((m.shape[1], m.shape[1], len(rot))) * (ppars.badmatch_penalty + 1)
    
    c_vp = 0
    #for t in range(1):
    for t in range(len(rot)):
        #theta = -rot[t] * np.pi / 180      # rotation of F2
        t_rot = time.time()
        theta = rot[t]
        theta_rad = theta * np.pi / 180     # np.deg2rad(theta) ?
        for ix in range(m.shape[1]):        # (z_id.shape[0]):
            t_x = time.time()
            for iy in range(m.shape[1]):    # (z_id.shape[0]):
                t_y = time.time()
                z = z_id[iy, ix]            # ??? [iy,ix] ??? strange...
                valid_point = mask_ij[iy, ix, t]
                print(iy, ix, t)
                if valid_point > 0:
                    print(f"Min val of s11: {np.min(s11)}, s12: {np.min(s12)}, s21: {np.min(s21)}, s22: {np.min(s22)}")
                    c_vp += 1
                    # print([iy, ix, t])
                    # check if line1 crosses the polygon2                  
                    intersections1, useful_lines_s11, useful_lines_s12, poly2_T, l1_T, ul1_T = \
                        line_poligon_intersect(z[::-1], -theta, poly2, [0, 0],  0, poly1, s11, s12, pars, extrapolate=False,
                        return_shapes=True)

                    # return intersections                    
                    useful_lines_alfa1 = alfa1[intersections1]  # no rotation here!
                    useful_lines_color1 = color1[intersections1]
                    useful_lines_cat1 = cat1[intersections1]
                    useful_lines_s11 = useful_lines_s11[intersections1]
                    useful_lines_s12 = useful_lines_s12[intersections1]

                    # check if line2 crosses the polygon1
                    intersections2, useful_lines_s21, useful_lines_s22, poly1_T, l2_T, ul2_T = \
                        line_poligon_intersect([0, 0], 0, poly1, z[::-1], -theta, poly2, s21, s22, pars, extrapolate=False,
                        return_shapes=True)
                    useful_lines_alfa2 = alfa2[intersections2] + theta_rad # the rotation!

                    useful_lines_color2 = color2[intersections2]
                    useful_lines_cat2 = cat2[intersections2]
                    useful_lines_s21 = useful_lines_s21[intersections2]
                    useful_lines_s22 = useful_lines_s22[intersections2]

                    n_lines_f1 = useful_lines_alfa1.shape[0]
                    n_lines_f2 = useful_lines_alfa2.shape[0]

                    if show == True:
                        plt.subplot(2,2,c_vp)
                        plt.plot(*poly1_T.boundary.xy, color='red')
                        for l1 in l1_T:
                            plt.plot(*l1.xy, color='orange')
                        for ul1 in ul1_T:
                            plt.plot(*ul1.xy, color='green', linewidth=3)
                        plt.plot(*poly2_T.boundary.xy, color='blue')
                        for l2 in l2_T:
                            plt.plot(*l2.xy, color='lightblue')
                        for ul2 in ul2_T:
                            plt.plot(*ul2.xy, color='green', linewidth=3)


                    if n_lines_f1 == 0 and n_lines_f2 == 0:
                        #tot_cost = ppars.max_dist * 2  
                        tot_cost = ppars.badmatch_penalty / 3                   # accept with some cost

                    elif (n_lines_f1 == 0 and n_lines_f2 > 0) or (n_lines_f1 > 0 and n_lines_f2 == 0):
                        n_lines = (np.max([n_lines_f1, n_lines_f2]))
                        tot_cost = ppars.mismatch_penalty * n_lines

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
                        #print([tot_cost])
                        print("#" * 50)
                        print(dist_matrix)
                        
                        # # penalty
                        penalty = np.abs(n_lines_f1 - n_lines_f2) * ppars.mismatch_penalty  # no matches penalty
                        tot_cost = (tot_cost + penalty)
                        tot_cost = tot_cost / np.max([n_lines_f1, n_lines_f2])  # normalize to all lines in the game
                        print(tot_cost)
                    if show == True:
                        plt.title(f'Cost: {tot_cost}\n(useful lines poly1: {n_lines_f1}, useful lines poly2: {n_lines_f2})')    
                        R_cost[iy, ix, t] = tot_cost
                if verbosity > 4:
                    print(f"comp on y took {(time.time()-t_y):.02f} seconds")
            if verbosity > 3:
                print(f"comp on x,y took {(time.time()-t_x):.02f} seconds")
        if verbosity > 2:
            print(f"comp on t = {t} (for all x,y) took {(time.time()-t_rot):.02f} seconds ({np.sum(mask_ij[:, :, t]>0)} valid values)")
    
    if show == True:
        plt.axis('equal')
        plt.show()
        pdb.set_trace()
    print(R_cost)
    R_cost[R_cost > ppars.badmatch_penalty] = ppars.badmatch_penalty
    len_unique = len(np.unique(R_cost))
    kmin_cut_val = np.sort(np.unique(R_cost))[::-1][-min(len_unique,ppars.k)]
    norm_R_cost = np.maximum(1 - R_cost / kmin_cut_val, 0)
    print(norm_R_cost)
    
    return norm_R_cost


def compute_line_based_CM_LAP(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, \
    color1, color2, cat1, cat2, mask_ij, ppars, verbosity=1, guglielmo=3):
    compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot))) 
    #for t in range(1):
    for t in range(len(rot)):
        #theta = -rot[t] * np.pi / 180      # rotation of F2
        t_rot = time.time()
        theta = rot[t]
        theta_rad = theta * np.pi / 180     # np.deg2rad(theta) ?
        for ix in range(m.shape[1]):        # (z_id.shape[0]):
            t_x = time.time()
            for iy in range(m.shape[1]):    # (z_id.shape[0]):
                t_y = time.time()
                z = z_id[iy, ix]            # ??? [iy,ix] ??? strange...
                valid_point = mask_ij[iy, ix, t]
                #print(iy, ix, t)
                if valid_point > 0:
                    # print([iy, ix, t])
                    # check if line1 crosses the polygon2                  
                    intersections1, useful_lines_s11, useful_lines_s12 = line_poligon_intersect(z[::-1], -theta, poly2, [0, 0],  0, poly1, s11, s12, ppars)

                    # return intersections                    
                    useful_lines_alfa1 = alfa1[intersections1]  # no rotation here!
                    useful_lines_color1 = color1[intersections1]
                    useful_lines_cat1 = cat1[intersections1]
                    useful_lines_s11 = useful_lines_s11[intersections1]
                    useful_lines_s12 = useful_lines_s12[intersections1]

                    # check if line2 crosses the polygon1
                    intersections2, useful_lines_s21, useful_lines_s22 = line_poligon_intersect([0, 0], 0, poly1, z[::-1], -theta, poly2, s21, s22, ppars)
                    useful_lines_alfa2 = alfa2[intersections2] + theta_rad # the rotation!

                    useful_lines_color2 = color2[intersections2]
                    useful_lines_cat2 = cat2[intersections2]
                    useful_lines_s21 = useful_lines_s21[intersections2]
                    useful_lines_s22 = useful_lines_s22[intersections2]

                    n_lines_f1 = useful_lines_alfa1.shape[0]
                    n_lines_f2 = useful_lines_alfa2.shape[0]

                    if n_lines_f1 == 0 and n_lines_f2 == 0:
                        #tot_cost = ppars.badmatch_penalty/2 # accept with some cost, guglielmo=3
                        #tot_cost = ppars.max_dist
                        tot_cost = ppars.badmatch_penalty * 0.9
                        tot_cost = ppars.badmatch_penalty / 3                   # accept with some cost


                    elif (n_lines_f1 == 0 and n_lines_f2 > 0) or (n_lines_f1 > 0 and n_lines_f2 == 0):
                        n_lines = (np.max([n_lines_f1, n_lines_f2]))
                        #tot_cost = ppars.mismatch_penalty*n_lines**2   ## it will be very high compatibility
                        tot_cost = ppars.badmatch_penalty * 0.9
                        tot_cost = ppars.mismatch_penalty * n_lines
                        
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
                        #print([tot_cost])
                        #print("#" * 50)
                        #print(dist_matrix)
                        
                        # # penalty
                        penalty = np.abs(n_lines_f1 - n_lines_f2) * ppars.mismatch_penalty  # no matches penalty
                        # tot_cost = tot_cost / np.min([n_lines_f1, n_lines_f2])  # normalize to all lines in the game
                        tot_cost = (tot_cost + penalty)
                        tot_cost = tot_cost / np.max([n_lines_f1, n_lines_f2])  # normalize to all lines in the game

                        # print(f"R[{iy},{ix},{t}] = {tot_cost} (penalty was {penalty}, n1={n_lines_f1}, n2={n_lines_f2})")
                        
                    compatibility_score = np.clip(ppars.badmatch_penalty - tot_cost, 0, ppars.badmatch_penalty)
                    compatibility_matrix[iy, ix, t] = compatibility_score
                if verbosity > 4:
                    print(f"comp on y took {(time.time()-t_y):.02f} seconds")
            if verbosity > 3:
                print(f"comp on x,y took {(time.time()-t_x):.02f} seconds")
        if verbosity > 2:
            print(f"comp on t = {t} (for all x,y) took {(time.time()-t_rot):.02f} seconds ({np.sum(mask_ij[:, :, t]>0)} valid values)")
    
    # plt.imshow(compatibility_matrix)
    # plt.show()
    # # plt.title()
    # breakpoint()
    # #print(R_cost)
    # R_cost[R_cost > ppars.badmatch_penalty] = ppars.badmatch_penalty
    # len_unique = len(np.unique(R_cost))
    # kmin_cut_val = np.sort(np.unique(R_cost))[::-1][-min(len_unique,ppars.k)]
    # norm_R_cost = np.maximum(1 - R_cost / kmin_cut_val, 0)
    # #print(norm_R_cost)
    
    return compatibility_matrix


# compute cost matrix NEW version
def compute_line_based_CM_LCI(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, \
    mask_ij, ppars, verbosity=1):
    """
    Compute the cost using the Line-Confidence-Importance method (LCI), which weights the contribution of each line 
    (positive or negative) using the confidence (at the moment binary) and the importance (the length of the line).
    """
    compatibility_matrix = np.zeros((m.shape[1], m.shape[1], len(rot)))

    # for t in range(1):
    for t in range(len(rot)):
        theta = rot[t]
        theta_rad = theta * np.pi / 180  # np.deg2rad(theta) ?
        for ix in range(m.shape[1]):  # (z_id.shape[0]):
            for iy in range(m.shape[1]):  # (z_id.shape[0]):
                z = z_id[iy, ix]
                valid_point = mask_ij[iy, ix, t]
                if valid_point > 0:
                    # check if line1 crosses the polygon2
                    intersections1, useful_lines_s11, useful_lines_s12 = line_poligon_intersect(z[::-1], -theta, poly2,
                                                                                                [0, 0], 0, poly1, s11, s12,
                                                                                                ppars)
                    # return intersections
                    useful_lines_alfa1 = alfa1[intersections1]  # no rotation here!
                    useful_lines_color1 = color1[intersections1]
                    useful_lines_cat1 = cat1[intersections1]
                    useful_lines_s11 = useful_lines_s11[intersections1]
                    useful_lines_s12 = useful_lines_s12[intersections1]

                    # check if line2 crosses the polygon1
                    intersections2, useful_lines_s21, useful_lines_s22 = line_poligon_intersect([0, 0], 0, poly1,
                                                                                                z[::-1], -theta, poly2, s21,
                                                                                                s22, ppars)
                    useful_lines_alfa2 = alfa2[intersections2] + theta_rad  # the rotation!
                    useful_lines_color2 = color2[intersections2]
                    useful_lines_cat2 = cat2[intersections2]
                    useful_lines_s21 = useful_lines_s21[intersections2]
                    useful_lines_s22 = useful_lines_s22[intersections2]

                    n_lines_f1 = useful_lines_alfa1.shape[0]
                    n_lines_f2 = useful_lines_alfa2.shape[0]

                    # 1. Lines Importance
                    line_importance_f1 = np.zeros(n_lines_f1)
                    for idx in range(n_lines_f1):
                        line_importance_f1[idx] = distance.euclidean(useful_lines_s11[idx], useful_lines_s12[idx])
                    line_importance_f1 /= np.sum(line_importance_f1)

                    line_importance_f2 = np.zeros(n_lines_f2)
                    for idx in range(n_lines_f2):
                        line_importance_f2[idx] = distance.euclidean(useful_lines_s21[idx], useful_lines_s22[idx])
                    line_importance_f2 /= np.sum(line_importance_f2)

                    ######################################
                    # 1. Gamma, Distance, Confidence
                    cont_conf_f1 = -1
                    cont_conf_f2 = -1

                    if n_lines_f1 > 0 and n_lines_f2 > 0:
                        gamma_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        dist_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        color_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        cat_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        for i in range(n_lines_f1):
                            for j in range(n_lines_f2):
                                ## NEW - check output !!!
                                color_matrix[i, j] = np.all(useful_lines_color1[i, :] == useful_lines_color2[j, :])
                                cat_matrix[i, j] = np.all(useful_lines_cat1[i] == useful_lines_cat2[j])
                                gamma = useful_lines_alfa1[i] - useful_lines_alfa2[j]
                                gamma_matrix[i, j] = np.abs(np.sin(gamma))

                                d1 = distance.euclidean(useful_lines_s11[i], useful_lines_s21[j])
                                d2 = distance.euclidean(useful_lines_s11[i], useful_lines_s22[j])
                                d3 = distance.euclidean(useful_lines_s12[i], useful_lines_s21[j])
                                d4 = distance.euclidean(useful_lines_s12[i], useful_lines_s22[j])

                                dist_matrix[i, j] = np.min([d1, d2, d3, d4])

                        thr_gamma = ppars.thr_coef
                        thr_dist = ppars.max_dist
                        cont_confidence = np.zeros((n_lines_f1, n_lines_f2)) - 1  # initially is NEGATIVE
                        cont_confidence[gamma_matrix < thr_gamma] = 1  # positive confidence to co-linear lines
                        cont_confidence[dist_matrix > thr_dist] = -1  # negative confidence to distant lines
                        # # new - check if works !!!
                        cont_confidence[cat_matrix < 1] = -1  # negative confidence color non matching

                        cont_conf_f1 = np.max(cont_confidence, 1)  # confidence vector (-1/1) for lines of A
                        cont_conf_f2 = np.max(cont_confidence, 0)  # confidence vector (-1/1) for lines of B

                    score_f1 = np.sum(cont_conf_f1 * line_importance_f1)
                    score_f2 = np.sum(cont_conf_f2 * line_importance_f2)

                    # sum of confident lines - sum of non-confident lines
                    if score_f1 > 0 and score_f2 > 0:
                        # if n_lines_f1 > 0 and n_lines_f2 > 0:
                        #     print(f"Gamma Matrix: {np.transpose(gamma_matrix)}")
                        #     print(f"Dist Matrix: {np.transpose(dist_matrix)}")
                        #     print(f"Category Matrix: {np.transpose(cat_matrix)}")
                        #     print(f"Conf Matrix: {np.transpose(cont_confidence)}")
                        # print(f"cost {score_f1 + score_f2} // cost for pieces {score_f1} and {score_f2} in {[iy, ix, t]}")
                        # print("\n\n")
                        compatibility_score = score_f1 + score_f2
                        if verbosity > 2:
                            print(f"cost {score_f1 + score_f2} // cost for pieces {score_f1} and {score_f2} in {[iy, ix, t]}")
                    else:
                        compatibility_score = 0 #score_f1 + score_f2
                    #ind = np.argpartition(line_importance_f2, -n_lines_f1)[-:]
                    #tot_cost=tot_cost/(np.sum(line_importance_f1)+np_sum(np.kmax)
                    compatibility_matrix[iy, ix, t] = compatibility_score

    #R_cost = np.maximum(R_cost, 0)
    return compatibility_matrix

# warning! moved to comp_utils.py!
# def compute_cost_wrapper(idx1, idx2, pieces, regions_mask, cmp_parameters, ppars, compatibility_type='lines', verbosity=1):
#     """
#     Wrapper for the cost computation, so that it can be called in one-line, 
#     making it easier to parallelize using joblib's Parallel (in comp_irregular.py) 

#     # shape branch
#     added a "compatibility_type" parameter which allows to control which compatibility to use:
#     shape, color, line, pattern.. 
#     """

#     (p, z_id, m, rot, line_matching_pars) = cmp_parameters
#     n = len(pieces)
    
#     if verbosity > 1:
#         print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")

#     if idx1 == idx2:
#         #print('idx == ')
#         R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1
#     else:
#         poly1 = pieces[idx1]['polygon']
#         poly2 = pieces[idx2]['polygon']
#         mask_ij = regions_mask[:, :, :, idx2, idx1]
#         candidate_values = np.sum(mask_ij > 0)
#         if compatibility_type == 'lines':
#             alfa1, r1, s11, s12, color1, cat1 = extract_from(pieces[idx1]['extracted_lines'])
#             alfa2, r2, s21, s22, color2, cat2 = extract_from(pieces[idx2]['extracted_lines'])
#             if len(alfa1) == 0 and len(alfa2) == 0:
#                 #print('no lines')
#                 R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + line_matching_pars.max_dist * 2
#             elif (len(alfa1) > 0 and len(alfa2) == 0) or (len(alfa1) == 0 and len(alfa2) > 0):
#                 #print('only one side with lines')
#                 R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + line_matching_pars.mismatch_penalty
#             else:
#                 #print('values!')
                
#                 t1 = time.time()
#                 if line_matching_pars.cmp_cost == 'DEBUG':
#                     print(f"Computing compatibility between Piece {idx1} and Piece {idx2}")
#                     if idx2 - idx1 == 1:
#                         plt.suptitle(f"COST between Piece {idx1} and Piece {idx2}", fontsize=32)
#                         R_cost = compute_cost_matrix_LAP_debug(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
#                                                     mask_ij, ppars, verbosity=verbosity, show=True)
#                     else:
#                         R_cost = compute_cost_matrix_LAP_debug(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
#                                                     mask_ij, ppars, verbosity=verbosity, show=False)
#                 elif line_matching_pars.cmp_cost == 'LAP':
#                     R_cost = compute_cost_matrix_LAP(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
#                                                     mask_ij, ppars, verbosity=verbosity)
#                 elif line_matching_pars.cmp_cost == 'LCI':
#                     R_cost = compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
#                                                             mask_ij, ppars, verbosity=verbosity)
#                 elif line_matching_pars.cmp_cost == 'LAP2':
#                     R_cost = compute_cost_matrix_LAP_v2(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
#                                                             mask_ij, ppars, verbosity=verbosity)
#                 elif line_matching_pars.cmp_cost == 'LAP3':
#                     R_cost = compute_cost_matrix_LAP_v3(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
#                                                             mask_ij, ppars, verbosity=verbosity)
#                 else:
#                     print('weird: using {line_matching_pars.cmp_cost} method, not known! We use `new` as we dont know what else to do! change --cmp_cost')
#                     R_cost = compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, color1, color2, cat1, cat2, line_matching_pars,
#                                                             mask_ij, ppars)
#                 if verbosity > 1:
#                     print(f"computed cost matrix for piece {idx1} ({len(alfa1)} lines) vs piece {idx2} ({len(alfa2)} lines): took {(time.time()-t1):.02f} seconds ({candidate_values:.1f} candidate values) ")
#                 #print(R_cost)
#         elif compatibility_type == 'shape':
#             breakpoint()
#             ids_to_score = np.where(mask_ij > 0)
#             R_cost = compute_SDF_cost_matrix(pieces[idx1], pieces[idx2], ids_to_score, cmp_parameters, ppars)

#         else: # other compatibilities!
#             print("\n" * 20)
#             print("=" * 50)
#             print("WARNING:")
#             print(f"Received: {compatibility_type} as compatibility_type")
#             print("NOT IMPLEMENTED YET, RETURNING JUST EMPTY MATRIX")
#             print("=" * 50)
#             print("\n" * 20)

#             R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))
        
#     return R_cost

# Convert polar coordinates to Cartesian coordinates and display the result
def polar2cartesian(image, angles, dists, show_image=False):

    line_length = max(image.shape[0], image.shape[1])
    if show_image:
        plt.imshow(image, cmap=cm.gray)
    lines = []
    for angle, dist in zip(angles, dists):
        (x1, y1) = dist * np.array([np.cos(angle), np.sin(angle)])

        slope = np.tan(angle + np.pi / 2)
        # Calculate the y-intercept (b)
        intercept = y1 - slope * x1

        # Generate x-values for the line
        x = np.linspace(x1, x1 + 1, line_length)  # Adjust the range as needed

        # Calculate y-values using the slope-intercept form
        y = slope * x + intercept

        # Get the coordinates of the end point
        x2, y2 = x[-1], y[-1]

        lines.append((x1, y1, x2, y2))
        if show_image:
            plt.axline((x1, y1), (x2,y2))
    if show_image:
        plt.tight_layout()
        plt.show()

    return lines

def cluster_lines_dbscan(image, angles, dists, epsilon, min_samples):
    line_points = polar2cartesian(image, angles, dists)
    #line_points_ = np.expand_dims(np.array(line_points), axis=1)

    # Convert line endpoints to a suitable format for DBSCAN
    points = np.array(line_points)

    # Apply DBSCAN to cluster line endpoints
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # Find the most representative line in each cluster
    unique_labels = np.unique(labels)
    centroid_lines = []

    for label in unique_labels:
        #cluster_lines = line_points[labels == label]

        cluster_indices = np.where(labels == label)[0]
        cluster_lines = [line_points[idx] for idx in cluster_indices]

        # Calculate the representative line as the median of the lines in the cluster
        mean_line = np.mean(cluster_lines, axis=0)
        centroid_lines.append(mean_line)
    return centroid_lines

def line_cart2pol(pt1, pt2):

    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    theta = np.arctan(-(x_diff/(y_diff + 10**-5)))
    rho = pt1[0] * np.cos(theta) + pt1[1] * np.sin(theta)
    rho2 = pt2[0] * np.cos(theta) + pt2[1] * np.sin(theta)
    #print("checkcart2pol", theta, rho, rho2)
    #pdb.set_trace()
    return rho, theta
    
def draw_hough_lines(img, lines):
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            pt1, pt2 = line_pol2cart(rho=rho, theta=theta)
            cv2.line(img, pt1, pt2, (255, 0, 0), 1, cv2.LINE_AA)
    return img

def draw_prob_hough_line(img, linesP):
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 1, cv2.LINE_AA)
    return img 

def display_unprocessed_hough_result(image,theta,d,h):

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]
    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for i, (angle, dist) in enumerate(zip(theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    plt.tight_layout()
    plt.show()
