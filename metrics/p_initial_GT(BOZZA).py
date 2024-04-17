from typing import Any

import scipy
from scipy.io import loadmat
import pdb
import numpy as np
import argparse
import matplotlib.pyplot as plt
from configs import unified_cfg as cfg
from configs import folder_names as fnames
import os
import json
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def main(args):
# def initialization_from_GT(R, args.dataset, args.puzzle, args.anchor, args.p_pts):

    anchor_id = 5     # default
    #no_rotations = R.shape[2]
    no_rotations = 4
    theta_step = 360/no_rotations

    dataset_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset)
    print(dataset_folder)
    puzzles = [args.puzzle]

    print(f"\nEvaluate solution for: {puzzles}\n")

    for puzzle in puzzles:
        puzzle_folder = os.path.join(dataset_folder, puzzle)
        general_files = os.listdir(puzzle_folder)
        # solution_folders = [sol_fld for sol_fld in general_files if "solution" in sol_fld]
        with open(os.path.join(puzzle_folder, "ground_truth.json"), 'r') as gtj:
            ground_truth = json.load(gtj)
        with open(os.path.join(puzzle_folder, f"parameters_{puzzle}.json"), 'r') as gtj:
            img_parameters = json.load(gtj)
        with open(os.path.join(puzzle_folder, f"compatibility_parameters.json"), 'r') as gtj:
            cmp_parameters = json.load(gtj)

        num_pcs = img_parameters['num_pieces']
        gt_coord = np.zeros([num_pcs, 3])
        for j in range(num_pcs):
            gt_coord[j, 2] = (+1) * ground_truth[f"piece_{j:04d}"]['rotation']
            gt_coord[j, 0:2] = (-1) * np.asarray(ground_truth[f"piece_{j:04d}"]['translation'][::-1])

        # 1. GT - shifted and rotated in PX (anchor is in ref.point = [0,0,0])
        anc_coord = gt_coord[anchor_id, 0:2]
        anc_rot = gt_coord[anchor_id, 2]
        gt_shift = np.zeros([num_pcs, 3])
        gt_shift[:, 0:2] = gt_coord[:, 0:2] - anc_coord
        gt_shift[:, 2] = gt_coord[:, 2] - anc_rot
        # print(f"GT non rotated in px :")
        # print(gt_shift)

        all_rot = gt_shift[:, 2]
        all_rot = np.where(all_rot < 0, all_rot + 360, all_rot)
        gt_shift[:, 2] = all_rot

        # 2. Translate to yxz space
        gt_yxz = np.zeros([num_pcs, 3])
        gt_yxz[:, 0:2] = np.round(gt_shift[:, 0:2] / cmp_parameters['xy_step'])
        gt_yxz[:, 2] = gt_shift[:, 2] / cmp_parameters['theta_step']
        # print(f"GT non rotated in YX :")
        # print(gt_yxz)

        # 3. Rotate GT according anchor rotation - CHECK!!!
        gt_rot = np.zeros([num_pcs, 3])
        gt_rot[:, 2] = gt_yxz[:, 2]
        if anc_rot == 0:
            gt_rot[:, 0:2] = gt_yxz[:, 0:2]
        else:
            if anc_rot == 90 or anc_rot == -270:
                gt_rot[:, 0] = gt_yxz[:, 1]   # y_new = +x
                gt_rot[:, 1] = -gt_yxz[:, 0]  # x_new = -y
            elif anc_rot == -90 or anc_rot == 270:
                gt_rot[:, 0] = -gt_yxz[:, 1]  # y_new = -x
                gt_rot[:, 1] = gt_yxz[:, 0]   # x_new = +y##
            elif anc_rot == 180 or anc_rot == -180:
                gt_rot[:, 0] = -gt_yxz[:, 0]  # y_new = -x
                gt_rot[:, 1] = -gt_yxz[:, 1]  # x_new = -y

        ##### NEW PART STARTS HERE !!!!!!
        # Shift ALL coord to the center of the Reconstruction plane !!!!

        # grid_size = 11
        grid_size = args.p_pts
        p = np.zeros((grid_size, grid_size, no_rotations, num_pcs))

        center_grid = round(grid_size/2)
        anc_position = [center_grid, center_grid, 0]
        probability_centers = np.zeros([num_pcs, 3])
        probability_centers[:, 0:2] = gt_rot[:, 0:2] + anc_position[0:2]
        probability_centers[:, 2] = gt_rot[:, 2]

        # inputs for probability distribution
        sigma_x = 1
        sigma_y = 1

        #sigma_z = 1
        #yy, xx, zz = np.mgrid[0:grid_size:1, 0:grid_size:1, 0:no_rotations:1]
        #pos = np.dstack((xx, yy, zz))
        #cov3 = [[sigma_y, 0, 0], [0, sigma_x, 0], [0, 0, sigma_z]]

        yy, xx = np.mgrid[0:grid_size:1, 0:grid_size:1]
        pos = np.dstack((yy, xx))
        cov2 = [[sigma_y, 0], [0, sigma_x]]

        for j in range(num_pcs):
            # mu3 = probability_centers[j, :]
            mu2 = probability_centers[j, 0:2]
            rv = multivariate_normal(mu2, cov2)
            p_norm_j = rv.pdf(pos)

            for t in range(no_rotations):
                p[:, :, t, j] = p_norm_j

        print(f'  ')
        print(f"GT shifted and rotated in YX :")


    print("#" * 50)
    print("FINISHED")

# def est_mult_gaus(X,mu,sigma):
#     p = np.ones((args.p_pts, args.p_pts) / (args.p_pts * args.p_pts)
#     m_size =  0.5*(args.p_pts-1)
#
#     m2, m1 = np.meshgrid(np.linspace(-1, 1, m_size), np.linspace(-1, 1, m_size))
#
#
#     Xs = np.arange(0, args.p_pts, 1)
#     Ys = np.arange(0, args.p_pts, 1)
#
#     sigma2 = np.diag(sigma)
#     X = (X-mu).T
#     p = 1/((2*np.pi)**(m/2)*np.linalg.det(sigma2)**(0.5))*np.exp(-0.5*np.sum(X.dot(np.linalg.pinv(sigma2))*X,axis=1))
#
#     return p



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('-d', '--dataset', type=str,
                        default='patterns_for_boundary_seg',
                        help='dataset folder')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')  # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('--puzzle', type=str, default='image_00000', help='puzzle folder')
    parser.add_argument('--p_pts', type=int, default=15, help='the size of the p matrix (it will be p_pts x p_pts)')
    args = parser.parse_args()

    main(args)
