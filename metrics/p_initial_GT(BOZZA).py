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


def main(args):
# def initialization_from_GT(R, args.dataset, args.puzzle, args.anchor, args.p_pts):

    anchor_id = 5     # default
    no_rotations = R.shape[2]
    no_patches = R.shape[3]

    theta_step = 360 / no_rotations

    dataset_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset)
    print(dataset_folder)
    if args.puzzle == '':
        puzzles = os.listdir(dataset_folder)
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(dataset_folder, puz)) is True]
    else:
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
        if num_pcs != no_patches:
            print('Error - number of patches is not coincide !!!')

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

        # 2. Translate to yxz space
        gt_yxz = np.zeros([num_pcs, 3])
        gt_yxz[:, 0:2] = np.round(gt_shift[:, 0:2] / cmp_parameters['xy_step'])
        gt_yxz[:, 2] = gt_shift[:, 2] / theta_step
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

        x0 = round(args.p_pts / 2)
        y0 = round(args.p_pts / 2)
        z0 = 0
        anc_position = [y0, x0, z0]
        probability_centers = np.zeros([num_pcs, 3])
        probability_centers[:, 0:2] = gt_rot[:, 0:2] + anc_position
        probability_centers[:, 2] = gt_rot[:, 2] + z0

        for jj in range(noPatches):
            y = new_anc[jj, 0]
            x = new_anc[jj, 1]
            z = new_anc[jj, 2]
            p[:, :, :, jj] = 0
            p[y, x, :, :] = 0
            p[y, x, z, jj] = 1


        p = np.ones((args.p_pts, args.p_pts, no_rotations, num_pcs)) / (args.p_pts * args.p_pts)  # uniform

        print(f'  ')
        print(f"GT shifted and rotated in YX :")
        print(gt_rot)

    print("#" * 50)
    print("FINISHED")

def est_mult_gaus(X,mu,sigma):
    p = np.ones((args.p_pts, args.p_pts) / (args.p_pts * args.p_pts)

    m2, m1 = np.meshgrid(np.linspace(-1, 1, m_size), np.linspace(-1, 1, m_size))

    Xs = np.arange(0, args.p_pts, 1)
    Ys = np.arange(0, args.p_pts, 1)

    sigma2 = np.diag(sigma)
    X = (X-mu).T
    p = 1/((2*np.pi)**(m/2)*np.linalg.det(sigma2)**(0.5))*np.exp(-0.5*np.sum(X.dot(np.linalg.pinv(sigma2))*X,axis=1))

    return p



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('-d', '--dataset', type=str,
                        default='synthetic_irregular_9_pieces_by_drawing_coloured_lines_28_02_2024',
                        help='dataset folder')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')  # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-p', '--puzzle', type=str, default='', help='puzzle folder')
    args = parser.parse_args()

    main(args)
