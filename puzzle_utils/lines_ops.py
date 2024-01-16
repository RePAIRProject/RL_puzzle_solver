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

class CfgParameters(dict):
    __getattr__ = dict.__getitem__

def calc_line_matching_parameters(parameters, cmp_cost='new'):
    lm_pars = CfgParameters()
    lm_pars['thr_coef'] = 0.08
    lm_pars['max_dist'] = 0.70*(parameters.xy_step)
    lm_pars['badmatch_penalty'] = lm_pars['max_dist'] * 5 / 3 # parameters.piece_size / 3 #?
    lm_pars['mismatch_penalty'] = lm_pars['max_dist'] * 4 / 3 # parameters.piece_size / 4 #?
    lm_pars['rmax'] = lm_pars['max_dist'] * 7 / 6
    lm_pars['cmp_cost'] = cmp_cost
    return lm_pars

def draw_lines(lines_dict, img_shape, thickness=1, color=255):
    angles, dists, p1s, p2s = extract_from(lines_dict)
    lines_img = np.zeros(shape=img_shape[:2], dtype=np.uint8)
    for p1, p2 in zip(p1s, p2s):
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
    return angles, dists, p1s, p2s

def line_poligon_intersect(z_p, theta_p, poly_p, z_l, theta_l, s1, s2, pars):
    # check if line crosses the polygon
    # z_p1 = [0,0],  z_l2 = z,
    # z_p2 = z,   z_l1 = [0,0],
    intersections = []
    useful_lines_s1 = []
    useful_lines_s2 = []
    piece_j_shape = poly_p.tolist() #shapely.polygons(poly_p)
    piece_j_rotate = rotate(piece_j_shape, theta_p, origin=[pars.p_hs, pars.p_hs])
    piece_j_trans = transform(piece_j_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_p)

    for (candidate_xy_start, candidate_xy_end) in zip(s1, s2):

        candidate_line_shapely0 = shapely.LineString((candidate_xy_start, candidate_xy_end))
        candidate_line_rotate = rotate(candidate_line_shapely0, theta_l, origin=[pars.p_hs, pars.p_hs])
        candidate_line_trans = transform(candidate_line_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_l)

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

    return intersections, np.array(useful_lines_s1), np.array(useful_lines_s2)


def compute_cost_matrix_LAP(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, lmp, mask_ij, pars, verbosity=1):
    # lmp is the old cfg (with the parameters)
    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))

    #for t in range(1):
    for t in range(len(rot)):
        #theta = -rot[t] * np.pi / 180  # rotation of F2
        t_rot = time.time()
        theta = rot[t]
        theta_rad = theta * np.pi / 180 # np.deg2rad(theta) ?
        for ix in range(m.shape[1]):        # (z_id.shape[0]):
            t_x = time.time()
            for iy in range(m.shape[1]):    # (z_id.shape[0]):
                t_y = time.time()
                z = z_id[iy, ix]            # ??? [iy,ix] ??? strange...
                valid_point = mask_ij[iy, ix, t]
                if valid_point > 0:
                    #print([iy, ix, t])

                    # check if line1 crosses the polygon2                  
                    intersections1, useful_lines_s11, useful_lines_s12 = line_poligon_intersect(z[::-1], -theta, poly2, [0, 0], 0, s11, s12, pars)

                    # return intersections                    
                    useful_lines_alfa1 = alfa1[intersections1] # no rotation here!
                    useful_lines_s11 = useful_lines_s11[intersections1]
                    useful_lines_s12 = useful_lines_s12[intersections1]

                    # check if line2 crosses the polygon1
                    intersections2, useful_lines_s21, useful_lines_s22 = line_poligon_intersect([0, 0], 0, poly1, z[::-1], -theta, s21, s22, pars)

                    useful_lines_alfa2 = alfa2[intersections2] + theta_rad # the rotation!
                    useful_lines_s21 = useful_lines_s21[intersections2]
                    useful_lines_s22 = useful_lines_s22[intersections2]

                    n_lines_f1 = useful_lines_alfa1.shape[0]
                    n_lines_f2 = useful_lines_alfa2.shape[0]

                    if n_lines_f1 == 0 and n_lines_f2 == 0:
                        tot_cost = lmp.max_dist * 2  # accept with some cost

                    elif (n_lines_f1 == 0 and n_lines_f2 > 0) or (n_lines_f1 > 0 and n_lines_f2 == 0):
                        n_lines = (np.max([n_lines_f1, n_lines_f2]))
                        tot_cost = lmp.mismatch_penalty * n_lines

                    else:
                        # Compute cost_matrix, LAP, penalty, normalize
                        dist_matrix0 = np.zeros((n_lines_f1, n_lines_f2))
                        dist_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        gamma_matrix = np.zeros((n_lines_f1, n_lines_f2))

                        for i in range(n_lines_f1):
                            for j in range(n_lines_f2):
                                gamma = useful_lines_alfa1[i] - useful_lines_alfa2[j]
                                gamma_matrix[i, j] = np.abs(np.sin(gamma))

                                d1 = distance.euclidean(useful_lines_s11[i], useful_lines_s21[j])
                                d2 = distance.euclidean(useful_lines_s11[i], useful_lines_s22[j])
                                d3 = distance.euclidean(useful_lines_s12[i], useful_lines_s21[j])
                                d4 = distance.euclidean(useful_lines_s12[i], useful_lines_s22[j])

                                dist_matrix[i, j] = np.min([d1, d2, d3, d4])

                        dist_matrix[gamma_matrix > lmp.thr_coef] = lmp.badmatch_penalty
                        dist_matrix[dist_matrix > lmp.max_dist] = lmp.badmatch_penalty

                        # # LAP
                        row_ind, col_ind = linear_sum_assignment(dist_matrix)
                        tot_cost = dist_matrix[row_ind, col_ind].sum()
                        #print([tot_cost])

                        # # penalty
                        penalty = np.abs(n_lines_f1 - n_lines_f2) * lmp.mismatch_penalty  # no matches penalty
                        tot_cost = (tot_cost + penalty)
                        tot_cost = tot_cost / np.max([n_lines_f1, n_lines_f2])  # normalize to all lines in the game

                    R_cost[iy, ix, t] = tot_cost
                
                #print(f"comp on y took {(time.time()-t_y):.02f} seconds")
            #print(f"comp on x,y took {(time.time()-t_x):.02f} seconds")
        if verbosity > 2:
            print(f"comp on t = {t} (for all x,y) took {(time.time()-t_rot):.02f} seconds ({np.sum(mask_ij[:, :, t]>0)} valid values)")

    return R_cost


# compute cost matrix NEW version
def compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, lmp,
                                   mask_ij, pars, verbosity=1):
    """
    Compute the cost using the Line-Confidence-Importance method (LCI), which weights the contribution of each line 
    (positive or negative) using the confidence (at the moment binary) and the importance (the length of the line).
    """
    # lmp is the old cfg (with the parameters)
    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))

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
                                                                                                [0, 0], 0, s11, s12,
                                                                                                pars)
                    # return intersections
                    useful_lines_alfa1 = alfa1[intersections1]  # no rotation here!
                    useful_lines_s11 = useful_lines_s11[intersections1]
                    useful_lines_s12 = useful_lines_s12[intersections1]

                    # check if line2 crosses the polygon1
                    intersections2, useful_lines_s21, useful_lines_s22 = line_poligon_intersect([0, 0], 0, poly1,
                                                                                                z[::-1], -theta, s21,
                                                                                                s22, pars)
                    useful_lines_alfa2 = alfa2[intersections2] + theta_rad  # the rotation!
                    useful_lines_s21 = useful_lines_s21[intersections2]
                    useful_lines_s22 = useful_lines_s22[intersections2]

                    n_lines_f1 = useful_lines_alfa1.shape[0]
                    n_lines_f2 = useful_lines_alfa2.shape[0]

                    # 1. Lines Importance
                    line_importance_f1 = np.zeros(n_lines_f1)
                    for idx in range(n_lines_f1):
                        line_importance_f1[idx] = distance.euclidean(useful_lines_s11[idx], useful_lines_s12[idx])

                    line_importance_f2 = np.zeros(n_lines_f2)
                    for idx in range(n_lines_f2):
                        line_importance_f2[idx] = distance.euclidean(useful_lines_s21[idx], useful_lines_s22[idx])

                    ######################################
                    # 1. Gamma, Distance, Confidence
                    cont_confidence = -1
                    cont_conf_f1 = -1
                    cont_conf_f2 = -1

                    if n_lines_f1 > 0 and n_lines_f2 > 0:
                        gamma_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        dist_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        for i in range(n_lines_f1):
                            for j in range(n_lines_f2):
                                gamma = useful_lines_alfa1[i] - useful_lines_alfa2[j]
                                gamma_matrix[i, j] = np.abs(np.sin(gamma))

                                d1 = distance.euclidean(useful_lines_s11[i], useful_lines_s21[j])
                                d2 = distance.euclidean(useful_lines_s11[i], useful_lines_s22[j])
                                d3 = distance.euclidean(useful_lines_s12[i], useful_lines_s21[j])
                                d4 = distance.euclidean(useful_lines_s12[i], useful_lines_s22[j])

                                dist_matrix[i, j] = np.min([d1, d2, d3, d4])

                        thr_gamma = 0.08  # lmp.thr_coef
                        thr_dist = 3  # lmp.max_dist
                        cont_confidence = np.zeros((n_lines_f1, n_lines_f2)) - 1  # initially is NEGATIVE
                        cont_confidence[gamma_matrix < thr_gamma] = 1  # positive confidence to co-linear lines
                        cont_confidence[dist_matrix > thr_dist] = -1  # negative confidence to distant lines

                        cont_conf_f1 = np.max(cont_confidence, 1)  # confidence vector (-1/1) for lines of A
                        cont_conf_f2 = np.max(cont_confidence, 0)  # confidence vector (-1/1) for lines of B

                    cost_f1 = np.sum(cont_conf_f1 * line_importance_f1)
                    cost_f2 = np.sum(cont_conf_f2 * line_importance_f2)

                      # sum of confident lines - sum of non-confident lines
                    if cost_f1 > 0 and cost_f2 > 0:
                        tot_cost = cost_f1 + cost_f2
                        if verbosity > 2:
                            print(f"cost for pieces {cost_f1} and {cost_f2} in {[iy, ix, t]}")

                    tot_cost = cost_f1 + cost_f2
                    R_cost[iy, ix, t] = tot_cost

    rrr = np.max(R_cost)
    if verbosity > 2:
        print(f"max R value {rrr}")
    R_cost = np.maximum(R_cost, 0)
    return R_cost


def compute_cost_wrapper(idx1, idx2, pieces, regions_mask, cmp_parameters, ppars, verbosity=1):
    """
    Wrapper for the cost computation, so that it can be called in one-line, making it easier to parallelize using joblib's Parallel (in comp_irregular.py) 
    """

    (p, z_id, m, rot, line_matching_pars) = cmp_parameters
    n = len(pieces)
    
    if verbosity > 1:
        print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")

    if idx1 == idx2:
        #print('idx == ')
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1
    else:
        alfa1, r1, s11, s12 = extract_from(pieces[idx1]['extracted_lines'])
        alfa2, r2, s21, s22 = extract_from(pieces[idx2]['extracted_lines'])

        if len(alfa1) == 0 and len(alfa2) == 0:
            #print('no lines')
            R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + line_matching_pars.max_dist * 2
        elif (len(alfa1) > 0 and len(alfa2) == 0) or (len(alfa1) == 0 and len(alfa2) > 0):
            #print('only one side with lines')
            R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + line_matching_pars.mismatch_penalty
        else:
            #print('values!')
            poly1 = pieces[idx1]['polygon']
            poly2 = pieces[idx2]['polygon']
            mask_ij = regions_mask[:, :, :, idx2, idx1]
            candidate_values = np.sum(mask_ij > 0)
            t1 = time.time()
            if line_matching_pars.cmp_cost == 'LAP':
                R_cost = compute_cost_matrix_LAP(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11,
                    s12, s21, s22, poly1, poly2, line_matching_pars, mask_ij, ppars, verbosity=verbosity)
            elif line_matching_pars.cmp_cost == 'LCI':
                R_cost = compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11,
                    s12, s21, s22, poly1, poly2, line_matching_pars, mask_ij, ppars, verbosity=verbosity)
            else:
                print('weird: using {line_matching_pars.cmp_cost} method, not known! We use `new` as we dont know what else to do! change --cmp_cost')
                R_cost = compute_cost_matrix_LCI_method(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11,
                    s12, s21, s22, poly1, poly2, line_matching_pars, mask_ij, ppars)

            if verbosity > 1:
                print(f"computed cost matrix for piece {idx1} ({len(alfa1)} lines) vs piece {idx2} ({len(alfa2)} lines): took {(time.time()-t1):.02f} seconds ({candidate_values:.1f} candidate values) ")
            #print(R_cost)
    return R_cost

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
