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
    dataset_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset)
    print(dataset_folder)
    if args.puzzle == '':
        puzzles = os.listdir(dataset_folder)
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(dataset_folder, puz)) is True]
    else:
        puzzles = [args.puzzle]

    puzzle_folder = os.path.join(dataset_folder, puzzle)
    general_files = os.listdir(puzzle_folder)
    # solution_folders = [sol_fld for sol_fld in general_files if "solution" in sol_fld]

    print(f"\nEvaluate solution for: {puzzles}\n")
    for puzzle in puzzles:
        with open(os.path.join(puzzle_folder, "ground_truth.json"), 'r') as gtj:
            ground_truth = json.load(gtj)
        with open(os.path.join(puzzle_folder, f"parameters_{puzzle}.json"), 'r') as gtj:
            img_parameters = json.load(gtj)
        with open(os.path.join(puzzle_folder, f"compatibility_parameters.json"), 'r') as gtj:
            cmp_parameters = json.load(gtj)

        anchor_id = 5
        no_rotations = 4
        theta_step = 360 / no_rotations
        anc_coord = gt_coord[anchor_id, 0:2]
        anc_rot = gt_coord[anchor_id, 2]

        num_pcs = img_parameters['num_pieces']
        gt_coord = np.zeros([num_pcs, 3])
        for j in range(num_pcs):
            gt_coord[j, 2] = (+1) * ground_truth[f"piece_{j:04d}"]['rotation']
            gt_coord[j, 0:2] = (-1) * np.asarray(ground_truth[f"piece_{j:04d}"]['translation'][::-1])

            # GT - shifted and rotated in PX (anchor is in ref.point = [0,0,0])
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
            if anc_rot == 0:
                gt_rot[:, 0:2] = gt_yxz[:, 0:2]
            else:
                if anc_rot == 90 or anc_rot == -270:
                    gt_rot[:, 0] = gt_yxz[:, 1]  # y_new = +x
                    gt_rot[:, 1] = -gt_yxz[:, 0]  # x_new = -y
                elif anc_rot == -90 or anc_rot == 270:
                    gt_rot[:, 0] = -gt_yxz[:, 1]  # y_new = -x
                    gt_rot[:, 1] = gt_yxz[:, 0]  # x_new = +y
                elif anc_rot == 180 or anc_rot == -180:
                    gt_rot[:, 0] = -gt_yxz[:, 0]  # y_new = -x
                    gt_rot[:, 1] = -gt_yxz[:, 1]  # x_new = -y
            gt_rot[:, 2] = gt_yxz[:, 2]
            print(f"  ")
            print(f"GT shifted and rotated in YX :")
            print(gt_rot)

    print("#" * 50)
    print("FINISHED")


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
