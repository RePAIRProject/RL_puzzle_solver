import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
import scipy.io
import os
import argparse
from configs import folder_names as fnames
import configs.repair_cfg as cfg


def read_info(folder, image):
    # read json file
    f_name = os.path.join(folder, image)
    with open(f_name, 'r') as file:
        data = json.load(file)
    if len(data) > 0:
        beta = np.array(data['angles'])
        R = np.array(data['dists'])
    else:
        beta = []
        R = []
    return beta, R


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


def compute_cost_matrix(p, z_id, m, rot, alfa1, alfa2, r1, r2):
    R_cost = np.zeros((m.shape[1], m.shape[1], len(rot)))

    for t in range(len(rot)):
        theta = -rot[t] * np.pi / 180  # rotation of F2
        for ix in range(m.shape[1]):  # (z_id.shape[0]):
            for iy in range(m.shape[1]):  # (z_id.shape[0]):

                z = z_id[ix, iy]
                n_lines_f1 = alfa1.shape[0]
                n_lines_f2 = alfa2.shape[0]
                cost_matrix = np.zeros((n_lines_f1, n_lines_f2))
                thr_matrix = np.zeros((n_lines_f1, n_lines_f2))

                for i in range(n_lines_f1):
                    for j in range(n_lines_f2):

                        ## translate reference point to the center
                        beta1, R_new1 = translation(alfa1[i], r1[i], p)
                        beta2, R_new2 = translation(alfa2[j], r2[j], p)
                        ## shift and rot line 2
                        beta3, R_new3 = translation(beta2 + theta, R_new2, -z)

                        ## dist from new point to line 1
                        R_new4 = dist_point_line(beta1, R_new1, z)

                        ## distance between 2 lines
                        gamma = beta1 - beta3
                        coef = np.abs(np.sin(gamma))

                        dist1 = np.sqrt(
                            (R_new1 ** 2 + R_new3 ** 2 - 2 * np.abs(R_new1 * R_new3) * np.cos(gamma)))
                        dist2 = np.sqrt(
                            (R_new2 ** 2 + R_new4 ** 2 - 2 * np.abs(R_new2 * R_new4) * np.cos(gamma)))

                        ## thresholding
                        if coef < cfg.thr_coef:
                            cost = (dist1 + dist2)
                        else:
                            cost = cfg.max_dist
                        cost_matrix[i, j] = cost
                        thr_matrix[i, j] = coef < cfg.thr_coef

                ## LAP
                if thr_matrix.sum() > 0:
                    row_ind, col_ind = linear_sum_assignment(
                        cost_matrix)
                    # tot_cost = cost_matrix[row_ind, col_ind].sum()  # original !
                    tot_cost = (cost_matrix[row_ind, col_ind] * thr_matrix[row_ind, col_ind]).sum() # threshold
                else:
                    tot_cost = cfg.max_dist
                R_cost[iy, ix, t] = tot_cost
    return R_cost


def visualize_matrices(rot_l, all_cost_matrix):
    n_fr = all_cost_matrix.shape[4]
    plt.figure()
    fig, axs = plt.subplots(n_fr, n_fr)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for fr1 in range(n_fr):
        for fr2 in range(n_fr):
            cost_f1f2 = all_cost_matrix[:, :, rot_l, fr2, fr1]
            axs[fr1, fr2].matshow(cost_f1f2, aspect='auto')
            axs[fr1, fr2].axis('off')
    plt.show()


# # MAIN
def main(args):
    # data load (line json, RM)
    data_folder = os.path.join(fnames.output_dir, args.puzzle, fnames.lines_output_name)
    hough_output = os.path.join(data_folder, args.method)
    pieces_files = os.listdir(hough_output)
    n = len(pieces_files)

    rm_name = 'RM_shape_repair_g28_101x101x24x10x10.mat'
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
        im1 = pieces_files[f1]  # read image 1
        alfa1, r1 = read_info(hough_output, im1)

        if len(alfa1) > 0:
            for f2 in range(n):  # select moving and rotating fragment
                if f1 == f2:
                    R_norm = np.zeros((m.shape[1], m.shape[1], len(rot))) - 1
                else:
                    im2 = pieces_files[f2]  # read image 2
                    alfa2, r2 = read_info(hough_output, im2)
                    if len(alfa2) == 0:
                        R_norm = np.zeros((m.shape[1], m.shape[1], len(rot)))
                    else:
                        # # compute matrix of matching costs
                        R_cost = compute_cost_matrix(p, z_id, m, rot, alfa1, alfa2, r1, r2)
                        All_cost[:, :, :, f2, f1] = R_cost
                        R_norm = np.maximum(1 - R_cost / cfg.rmax, 0)
                All_norm_cost[:, :, :, f2, f1] = R_norm

    # # apply region masks
    neg_reg = np.array(np.where(R_mask < 0, -1, 0))
    R_line = (All_norm_cost * R_mask + neg_reg)
    R_line = R_line * 2
    R_line[R_line < 0] = -0.5
    for jj in range(n):
        R_line[:, :, :, jj, jj] = -1

    # # visualize compatibility matrices
    # TO DO - add flag for visualization !!!
    for rot_layer in [0, 6]:
        # visualize_matrices(rot_layer, All_cost)
        visualize_matrices(rot_layer, All_norm_cost)
        visualize_matrices(rot_layer, R_line)

    # plt.figure()
    # C = All_norm_cost[:, :, 0, 0, 9]
    # plt.imshow(C, aspect='auto')
    # plt.show()

    # save output
    output_folder = os.path.join(fnames.output_dir, args.puzzle, fnames.cm_output_name)
    filename = f'{output_folder}\\CM_lines_{args.method}'
    scipy.io.savemat(f'{filename}.mat', R_line)

    return All_cost, All_norm_cost, R_line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('--puzzle', type=str, default='wikiart_kuroda_4x4', help='puzzle folder') # repair_g28, wikiart_kuroda_4x4
    parser.add_argument('--method', type=str, default='FLD', help='method line detection')  # Hough, FLD

    args = parser.parse_args()

    # main(args)
    All_cost, All_norm_cost, R_line = main(args)
