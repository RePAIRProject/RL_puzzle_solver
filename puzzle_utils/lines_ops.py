from skimage.transform import hough_line
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib import cm
import shapely
import pdb 
import math 
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
#from itertools import compress
import time 

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
    piece_j_shape = poly_p.tolist() #shapely.polygons(poly_p)
    piece_j_rotate = shapely.affinity.rotate(piece_j_shape, theta_p, origin=[pars.p_hs, pars.p_hs])
    piece_j_trans = shapely.transform(piece_j_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_p)

    for (candidate_xy_start, candidate_xy_end) in zip(s1, s2):

        candidate_line_shapely0 = shapely.LineString((candidate_xy_start, candidate_xy_end))
        candidate_line_rotate = shapely.affinity.rotate(candidate_line_shapely0, theta_l, origin=[pars.p_hs, pars.p_hs])
        candidate_line_trans = shapely.transform(candidate_line_rotate, lambda x: x - [pars.p_hs, pars.p_hs] + z_l)

        # if shapely.is_empty(shapely.intersection(candidate_line_shapely.buffer(pars.border_tolerance), piece_j_shape.buffer(pars.border_tolerance))):
        if shapely.is_empty(shapely.intersection(candidate_line_trans, piece_j_trans.buffer(pars.border_tolerance))):
            intersections.append(False)
        else:
            intersections.append(True)
    return intersections


def compute_cost_matrix(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, poly1, poly2, cfg, mask_ij, pars):
    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))

    #for t in range(1):
    for t in range(len(rot)):
        #theta = -rot[t] * np.pi / 180  # rotation of F2
        theta = rot[t]
        for ix in range(m.shape[1]):        # (z_id.shape[0]):
            for iy in range(m.shape[1]):    # (z_id.shape[0]):
                z = z_id[ix, iy]            # ??? [iy,ix] ??? strange...

                valid_point = mask_ij[iy, ix, t]
                if valid_point > 0:
                    #print([iy, ix, t])

                    # check if line1 crosses the polygon2
                    # intersections1 = line_poligon_intersec(z, [0, 0], s11, s12, cfg)  # z_p2 = z,   z_l1 = [0,0]
                    
                    intersections1 = line_poligon_intersect(z, theta, poly2, [0, 0], 0, s11, s12, pars)
                    

                    # return intersections                    
                    useful_lines_alfa1 = alfa1[intersections1] #list(compress(alfa1, intersections1)) #
                    useful_lines_rho1 = r1[intersections1] #list(compress(r1, intersections1)) #
                    useful_lines_s11 = np.clip(s11[intersections1], 0, pars.piece_size)
                    useful_lines_s12 = np.clip(s12[intersections1], 0, pars.piece_size)

                    # check if line2 crosses the polygon1
                    # intersections2 = line_poligon_intersec([0, 0], z, s21, s22, cfg)  # z_p1 = [0,0],  z_l2 = z
                    intersections2 = line_poligon_intersect([0, 0], 0, poly1, z, theta, s21, s22, pars)

                    useful_lines_alfa2 = alfa2[intersections2] #list(compress(alfa2, intersections1)) #alfa2[intersections2]
                    useful_lines_rho2 = r2[intersections2] #list(compress(r2, intersections1)) #r2[intersections2]
                    useful_lines_s21 = np.clip(s21[intersections2], 0, pars.piece_size)
                    useful_lines_s22 = np.clip(s22[intersections2], 0, pars.piece_size)

                    n_lines_f1 = useful_lines_alfa1.shape[0]
                    n_lines_f2 = useful_lines_alfa2.shape[0]

                    if n_lines_f1 == 0 and n_lines_f2 == 0:
                        tot_cost = cfg.max_dist * 2  # accept with some cost

                    elif (n_lines_f1 == 0 and n_lines_f2 > 0) or (n_lines_f1 > 0 and n_lines_f2 == 0):
                        n_lines = (np.max([n_lines_f1, n_lines_f2]))
                        tot_cost = cfg.mismatch_penalty * n_lines

                    else:
                        # Compute cost_matrix, LAP, penalty, normalize
                        dist_matrix0 = np.zeros((n_lines_f1, n_lines_f2))
                        dist_matrix = np.zeros((n_lines_f1, n_lines_f2))
                        gamma_matrix = np.zeros((n_lines_f1, n_lines_f2))

                        useful_lines_s21 = useful_lines_s21 + z
                        useful_lines_s22 = useful_lines_s22 + z

                        for i in range(n_lines_f1):
                            for j in range(n_lines_f2):
                                gamma = useful_lines_alfa1[i] - useful_lines_alfa2[j]
                                gamma_matrix[i, j] = np.abs(np.sin(gamma))

                                d1 = distance.euclidean(useful_lines_s11[i], useful_lines_s21[j])
                                d2 = distance.euclidean(useful_lines_s11[i], useful_lines_s22[j])
                                d3 = distance.euclidean(useful_lines_s12[i], useful_lines_s21[j])
                                d4 = distance.euclidean(useful_lines_s12[i], useful_lines_s22[j])

                                dist_matrix0[i, j] = np.min([d1, d2, d3, d4])

                        dist_matrix[gamma_matrix > cfg.thr_coef] = cfg.badmatch_penalty
                        dist_matrix[dist_matrix0 > cfg.max_dist] = cfg.badmatch_penalty
                        # dist_matrix[dist_matrix0 > cfg.badmatch_penalty] = cfg.badmatch_penalty

                        # # LAP
                        row_ind, col_ind = linear_sum_assignment(dist_matrix)
                        tot_cost = dist_matrix[row_ind, col_ind].sum()
                        #print([tot_cost])

                        # # penalty
                        penalty = np.abs(n_lines_f1 - n_lines_f2) * cfg.mismatch_penalty  # no matches penalty
                        tot_cost = (tot_cost + penalty)
                        tot_cost = tot_cost / np.max([n_lines_f1, n_lines_f2])  # normalize to all lines in the game

                    R_cost[iy, ix, t] = tot_cost

    return R_cost

def compute_cost_matrix_LAP(idx1, idx2, pieces, regions_mask, cmp_parameters, ppars, verbose=True):
    """
    Wrapper for the cost computation, so that it can be called in one-line, making it easier to parallelize using joblib's Parallel (in comp_irregular.py) 
    """

    (p, z_id, m, rot, cfg) = cmp_parameters
    n = len(pieces)
    
    if verbose is True:
        print(f"Computing cost for pieces {idx1:>2} and {idx2:>2}")

    if idx1 == idx2:
        #print('idx == ')
        R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1
    else:
        alfa1, r1, s11, s12 = extract_from(pieces[idx1]['extracted_lines'])
        alfa2, r2, s21, s22 = extract_from(pieces[idx2]['extracted_lines'])

        if len(alfa1) == 0 and len(alfa2) == 0:
            #print('no lines')
            R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + cfg.max_dist * 2
        elif (len(alfa1) > 0 and len(alfa2) == 0) or (len(alfa1) == 0 and len(alfa2) > 0):
            #print('only one side with lines')
            R_cost = np.zeros((m.shape[1], m.shape[1], len(rot))) + cfg.mismatch_penalty
        else:
            #print('values!')
            poly1 = pieces[idx1]['polygon']
            poly2 = pieces[idx2]['polygon']
            mask_ij = regions_mask[:, :, :, idx2, idx1]
            R_cost = compute_cost_matrix(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11,
                                        s12, s21, s22, poly1, poly2, cfg, mask_ij, ppars)
            #print(R_cost)
    return R_cost

# def polar_cartesian_check():
# print("check polar/cart")
# print("polar")
# print(angles, dists)
# print('\n')
# cartesians = polar2cartesian(seg_img, angles, dists)
# print('cartesian')
# print(cartesians)
# print('\n\n')
# for j, cartline in enumerate(cartesians):
#     #cartline = line_pol2cart(angle, dist)
#     print(f'\noriginal polar: theta={angles[j]:.3f}, rho={dists[j]:.3f}')
#     print(f'converted to cartesian: pt1 = ({cartline[0]:.3f}, {cartline[1]:.3f}), pt2 = ({cartline[2]:.3f}, {cartline[3]:.3f})')
#     rho, theta = line_cart2pol(cartline[0:2], cartline[2:4])
#     print(f'polar: theta={theta:.3f}, rho={rho:.3f}')

# print('\n\n')
# pdb.set_trace()

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
