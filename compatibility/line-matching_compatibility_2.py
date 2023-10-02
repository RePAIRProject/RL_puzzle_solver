
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
import scipy.io
import os
from configs import folder_names as fnames
import shapely
import configs.puzzle_from_image_cfg_8 as cfg
import pdb 

def read_info(folder, image):
    # read json file
    f_name = os.path.join(folder, image)
    with open(f_name, 'r') as file:
        data = json.load(file)
    if len(data) > 0:
        beta = np.array(data['angles'])
        R = np.array(data['dists'])
        s1 = np.array(data['p1s'])
        s2 = np.array(data['p2s'])
        b1 = np.array(data['b1s'])
        b2 = np.array(data['b2s'])
    else:
        beta = []
        R = []
        s1 = s2 = b1 = b2 = []

    return beta, R, s1, s2, b1, b2


def translation(beta, radius, point):
    # # a*x + b*y + c = 0  - initial equation of the line
    a = np.cos(beta)
    b = np.sin(beta)
    c = -radius

    # if b == 0:  # if beta 90°
    #     y = np.linspace(-2000, 2000)
    #     x = -c/a * np.ones_like(y)
    # else:
    #     x = np.linspace(-2000, 2000)
    #     y = 1/b * (-a*x - c)

    # # a*(x+point(x)) + b*(y+point(y)) + c = 0 -  equation of translated line
    c_new = (a * point[0] + b * point[1] + c)
    if b == 0:
        # y_new = np.linspace(-2000, 2000)
        # x = -c_new/a * np.ones_like(y_new)
        beta_new = beta
        R_new = -c_new
    else:
        # x = np.linspace(-2000, 2000)
        # y_new = 1/b * (-a*x - c_new)
        # # sign of angle
        if ((1 / b * (-a * point[0] - c) > point[1]) and (1 / b * (- c) > 0)) or (
                (1 / b * (-a * point[0] - c) < point[1]) and (1 / b * (- c) < 0)):
            beta_new = beta
            R_new = -c_new
        else:
            beta_new = (beta + np.pi) % (2 * np.pi)
            R_new = c_new
    return beta_new, R_new  # , x, y, y_new  à


def dist_point_line(beta, radius, point):
    a = np.cos(beta)
    b = np.sin(beta)
    c = -radius
    R_new = abs(a * point[0] + b * point[1] + c) / np.sqrt(a ** 2 + b ** 2)
    return R_new


def line_poligon_intersec(z_p, z_l, s1, s2, cfg):
    # check if line crosses the polygon
    # z_p1 = [0,0],  z_l2 = z,
    # z_p2 = z,   z_l1 = [0,0],
    intersections = []
    piece_j_shape = shapely.box(z_p[0] - cfg.p_hs, z_p[1] - cfg.p_hs, z_p[0] + cfg.p_hs, z_p[1] + cfg.p_hs)

    for (candidate_xy_start, candidate_xy_end) in zip(s1, s2):

        candidate_line_shapely0 = shapely.LineString((candidate_xy_start, candidate_xy_end))
        candidate_line_shapely = shapely.transform(candidate_line_shapely0, lambda x: x - [cfg.p_hs, cfg.p_hs] + z_l)

        # if shapely.is_empty(shapely.intersection(candidate_line_shapely, piece_j_shape)):
        # if shapely.is_empty(shapely.intersection(candidate_line_shapely.buffer(cfg.border_tolerance), piece_j_shape.buffer(cfg.border_tolerance))):
        if shapely.is_empty(shapely.intersection(candidate_line_shapely, piece_j_shape.buffer(cfg.border_tolerance))):
            intersections.append(False)
        else:
            intersections.append(True)

    return intersections


def compute_cost_matrix(p, z_id, m, rot, alfa1, alfa2,  r1, r2, s11, s12, s21, s22, b11, b12, b21, b22, cfg):
    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))

    for t in range(len(rot)):
        theta = -rot[t] * np.pi / 180  # rotation of F2
        for ix in range(m.shape[1]):  # (z_id.shape[0]):
            for iy in range(m.shape[1]):  # (z_id.shape[0]):
                z = z_id[ix, iy]

                # check if line1 crosses the polygon2
                intersections1 = line_poligon_intersec(z, [0,0], s11, s12, cfg) # z_p2 = z,   z_l1 = [0,0]
                ####################
                # intersections = []
                # piece_j_shape = shapely.box(z[0]-cfg.p_hs, z[1]-cfg.p_hs, z[0]+cfg.p_hs, z[1]+cfg.p_hs)
                #
                # for (candidate_xy_start, candidate_xy_end) in zip(s11, s12):

                #     candidate_line_shapely0 = shapely.LineString((candidate_xy_start, candidate_xy_end))
                #     candidate_line_shapely = shapely.transform(candidate_line_shapely0, lambda x: x-cfg.p_hs)
                #
                #     # if shapely.is_empty(shapely.intersection(candidate_line_shapely, piece_j_shape)):
                #     if shapely.is_empty(shapely.intersection(candidate_line_shapely, piece_j_shape.buffer(cfg.border_tolerance))):
                #     #if shapely.is_empty(shapely.intersection(candidate_line_shapely.buffer(cfg.border_tolerance), piece_j_shape.buffer(cfg.border_tolerance))):
                #         intersections.append(False)
                #     else:
                #         intersections.append(True)
                #
                # return intersections
                useful_lines_alfa1 = alfa1[intersections1]
                useful_lines_rho1 = r1[intersections1]

                # check if line2 crosses the polygon1
                intersections2 = line_poligon_intersec([0, 0], z, s21, s22, cfg)  # z_p1 = [0,0],  z_l2 = z
                ####################
                # intersections = []
                # ## buffer
                # piece_i_shape = shapely.box(0 - cfg.p_hs, 0 - cfg.p_hs, 0 + cfg.p_hs, 0 + cfg.p_hs)
                # for (candidate_xy_start, candidate_xy_end) in zip(s21, s22):
                #     candidate_line_shapely0 = shapely.LineString((candidate_xy_start, candidate_xy_end))
                #     candidate_line_shapely = shapely.transform(candidate_line_shapely0, lambda x : x-[cfg.p_hs, cfg.p_hs]+z)

                #     # check if line intersect
                #     #if shapely.is_empty(shapely.intersection(candidate_line_shapely, piece_i_shape)):
                #     if shapely.is_empty(shapely.intersection(candidate_line_shapely,
                #                                              piece_i_shape.buffer(cfg.border_tolerance))):
                #     #if shapely.is_empty(shapely.intersection(candidate_line_shapely.buffer(cfg.border_tolerance), piece_i_shape.buffer(cfg.border_tolerance))):
                #         intersections.append(False)
                #     else:
                #         intersections.append(True)
                useful_lines_alfa2 = alfa2[intersections2]
                useful_lines_rho2 = r2[intersections2]

                n_lines_f1 = useful_lines_alfa1.shape[0]
                n_lines_f2 = useful_lines_alfa2.shape[0]

                #n_lines = (np.min([n_lines_f1, n_lines_f2]))
                if n_lines_f1 == 0 and n_lines_f2 == 0:
                    #tot_cost = 0
                    tot_cost = cfg.mismatch_penalty

                ## INCLUDE THIS PART
                elif (n_lines_f1 == 0 and n_lines_f2 > 0) or (n_lines_f1 > 0 and n_lines_f2 == 0):
                    n_lines = (np.max([n_lines_f1, n_lines_f2]))
                    tot_cost = cfg.mismatch_penalty * n_lines  # ** cfg.mismatch_penalty_exp
                    # tot_cost = tot_cost/ (n_lines+1)
                    # tot_cost = cfg.max_dist

                else:

                    ## Compute cost_matrix, LAP, penalty, normalize
                    cost_matrix = np.zeros((n_lines_f1, n_lines_f2))

                    for i in range(n_lines_f1):
                        for j in range(n_lines_f2):
                            # translate reference point to the center
                            beta1, R_new1 = translation(useful_lines_alfa1[i], useful_lines_rho1[i], p)
                            beta2, R_new2 = translation(useful_lines_alfa2[j], useful_lines_rho2[j], p)
                            # shift and rot line 2
                            beta3, R_new3 = translation(beta2 + theta, R_new2, -z)

                            # dist from new point to line 1
                            R_new4 = dist_point_line(beta1, R_new1, z)

                            # distance between 2 lines
                            gamma = beta1 - beta3
                            coef = np.abs(np.sin(gamma))

                            dist1 = np.sqrt(
                                (R_new1 ** 2 + R_new3 ** 2 - 2 * np.abs(R_new1 * R_new3) * np.cos(gamma)))
                            dist2 = np.sqrt(
                                (R_new2 ** 2 + R_new4 ** 2 - 2 * np.abs(R_new2 * R_new4) * np.cos(gamma)))
                            cost0 = (dist1 + dist2)

                            # thresholding
                            if coef < cfg.thr_coef:
                                cost = np.min([cost0, cfg.max_dist])
                            else:
                                cost = cfg.max_dist
                            cost_matrix[i, j] = cost

                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    # tot_cost = cost_matrix[row_ind, col_ind].mean()
                    tot_cost = cost_matrix[row_ind, col_ind].sum()

                    # # penalty
                    # penalty = cfg.mismatch_penalty * np.abs(n_lines_f1 - n_lines_f2) ** cfg.mismatch_penalty_exp
                    penalty = np.abs(n_lines_f1-n_lines_f2)*cfg.mismatch_penalty
                    tot_cost = (tot_cost + penalty)
                    # tot_cost = tot_cost / np.min([n_lines_f1, n_lines_f2])
                    tot_cost = tot_cost / np.max([n_lines_f1, n_lines_f2])

                R_cost[iy, ix, t] = tot_cost

    return R_cost


def visualize_matrices(rot_l, all_cost_matrix, file_name):
    n_fr = all_cost_matrix.shape[4]
    #_min, _max = np.amin(n_fr), np.amax(n_fr)
    plt.figure()
    fig, axs = plt.subplots(n_fr, n_fr, figsize=(100, 100))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for fr1 in range(n_fr):
        for fr2 in range(n_fr):
            cost_f1f2 = all_cost_matrix[:, :, rot_l, fr2, fr1]
            # axs[fr1, fr2].matshow(cost_f1f2, aspect='auto')
            axs[fr1, fr2].matshow(cost_f1f2, aspect='auto', vmin=-0.5, vmax=2)
            axs[fr1, fr2].axis('off')
            axs[fr1, fr2].autoscale(False)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    # plt.show()


# MAIN
def main(args):

    # data load (line json, RM)
    data_folder = os.path.join(f"{fnames.output_dir}_{args.pieces}x{args.pieces}", args.dataset)
    hough_output = os.path.join(data_folder, args.puzzle, fnames.lines_output_name, args.method)
    pieces_files = os.listdir(hough_output)
    json_files = [piece_file for piece_file in pieces_files if piece_file[-4:] == 'json']
    n = len(json_files)
    if args.penalty > 0:
        cfg.mismatch_penalty = args.penalty
    # rm_name = 'RM_shape_repair_g28_101x101x24x10x10.mat'
    rm_name = f'RM_{args.dataset}.mat'
    mat = scipy.io.loadmat(os.path.join(data_folder, rm_name))
    R_mask = mat['RM']

    # xy_grid_points
    p = [cfg.p_hs, cfg.p_hs]     # center of piece [125,125] - ref.point for lines
    m_size = cfg.xy_grid_points  # 101X101 grid
    m = np.zeros((m_size, m_size, 2))
    m2, m1 = np.meshgrid(np.linspace(-1, 1, m_size), np.linspace(-1, 1, m_size))
    m[:, :, 0] = m1
    m[:, :, 1] = m2

    z_rad = cfg.pairwise_comp_range // 2
    z_id = m * z_rad

    ang = cfg.theta_step
    rot = np.arange(0, 360 - ang + 1, ang)

    All_cost = np.zeros((m.shape[1], m.shape[1], len(rot), n, n))
    All_norm_cost = np.zeros((m.shape[1], m.shape[1], len(rot), n, n))

    for f1 in range(n):  # select fixed fragment
        for f2 in range(n):  # select moving and rotating fragment
            if f1 == f2:
                R_norm = np.zeros((m.shape[1], m.shape[1], len(rot)))-1
            else:
                im1 = json_files[f1]  # read image 1
                alfa1, r1, s11, s12, b11, b12 = read_info(hough_output, im1)

                im2 = json_files[f2]  # read image 2
                alfa2, r2, s21, s22, b21, b22 = read_info(hough_output, im2)

                if len(alfa1) == 0 and len(alfa2) == 0:
                    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))+cfg.mismatch_penalty
                else:
                    R_cost = compute_cost_matrix(p, z_id, m, rot, alfa1, alfa2, r1, r2, s11, s12, s21, s22, b11, b12,
                                                 b21, b22, cfg)
                All_cost[:, :, :, f2, f1] = R_cost
                R_norm = np.maximum(1 - R_cost / cfg.rmax, 0)

                # if len(alfa2) == 0:
                #     R_norm = np.zeros((m.shape[1], m.shape[1], len(rot)))
                # else:
                #     # # compute matrix of matching costs
                #     R_cost = compute_cost_matrix(p, z_id, m, rot, alfa1, alfa2,  r1, r2, s11, s12, s21, s22, b11, b12, b21, b22, cfg)
                #     All_cost[:, :, :, f2, f1] = R_cost
                #     R_norm = np.maximum(1 - R_cost / cfg.rmax, 0)

            All_norm_cost[:, :, :, f2, f1] = R_norm

    # apply region masks
    neg_reg = np.array(np.where(R_mask < 0, -1, 0))
    R_line = (All_norm_cost * R_mask + neg_reg)
    R_line = R_line * 2
    R_line[R_line < 0] = -1  # -0.5  #
    for jj in range(n):
        R_line[:, :, :, jj, jj] = -1

    # save output
    output_folder = os.path.join(fnames.output_dir, args.dataset, args.puzzle, f"{fnames.cm_output_name}_{cfg.num_patches_side}x{cfg.num_patches_side}")
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f'CM_lines_{args.method}_p{cfg.mismatch_penalty}')
    mdic = {"R_line": R_line, "label": "label"}
    scipy.io.savemat(f'{filename}.mat', mdic)
    np.save(filename, R_line)

    # visualize compatibility matrices
    for rot_layer in [0]:
        file_vis_name = os.path.join(output_folder, f'CM_image_rot{rot_layer}_p{cfg.mismatch_penalty}')
        #visualize_matrices(rot_layer, All_cost)
        #visualize_matrices(rot_layer, All_norm_cost)
        visualize_matrices(rot_layer, R_line, file_vis_name)


    return All_cost, All_norm_cost, R_line


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('--dataset', type=str, default='manual_lines', help='dataset folder')   # repair, wikiart, manual_lines, architecture
    parser.add_argument('--puzzle', type=str, default='lines1', help='puzzle folder')           # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('--method', type=str, default='deeplsd', help='method line detection')  # Hough, FLD
    parser.add_argument('--penalty', type=int, default=-1, help='penalty (leave -1 to use the one from the config file)')
    parser.add_argument('--pieces', type=int, default=8, help='number of pieces (per side)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    args = parser.parse_args()
    # if args.dataset == 'repair':
    #     import configs.puzzle_from_fragments_cfg as cfg
    # else:
    #     import configs.puzzle_from_image_cfg as cfg

    # main(args)
    All_cost, All_norm_cost, R_line = main(args)


    # ## poly
    # top_left = sg.Point2(z[0] - cfg.p_hs, z[1] + cfg.p_hs)
    # top_right = sg.Point2(z[0] + cfg.p_hs, z[1] + cfg.p_hs)
    # bottom_left = sg.Point2(z[0] - cfg.p_hs, z[1] - cfg.p_hs)
    # bottom_right = sg.Point2(z[0] + cfg.p_hs, z[1] - cfg.p_hs)
    #
    # poly = sg.Polygon([top_left, top_right, bottom_right, bottom_left])
    # for (candidate_xy_start, candidate_xy_end, b_start, b_end, alfa, radius) in zip(s11, s12, b11, b12, alfa1, r1):
    #     a = sg.Point2(candidate_xy_start[0], candidate_xy_start[1])
    #     b = sg.Point2(candidate_xy_end[0], candidate_xy_end[1])
    #     if [b_start, b_end] == [1, 0]:  # valori invertite... 1
    #         r = sg.Ray2(a, a - b)
    #     elif [b_start, b_end] == [0, 1]:
    #         r = sg.Ray2(b, b - a)
    #     else:  # [0, 0]
    #         r = sg.Line2(np.cos(alfa), np.sin(alfa), -radius)
    #
    #     ## transform
    #     t_mat = np.eye(3)
    #     t_mat[0, 2] = z[0]
    #     t_mat[1, 2] = z[1]
    #     #r = r.transform(t_mat)
    #     i = sg.intersection(r, poly)
    #
    #     draw(poly)
    #     draw(r)
    #     draw(i)
    #     plt.show()
    #################


