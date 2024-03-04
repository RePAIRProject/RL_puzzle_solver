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


def main(args):
    dataset_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset)
    print(dataset_folder)
    if args.puzzle == '':
        puzzles = os.listdir(dataset_folder)
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(dataset_folder, puz)) is True]
    else:
        puzzles = [args.puzzle]

    print(f"\nEvaluate solution for: {puzzles}\n")
    for puzzle in puzzles:

        print("\n\n")
        print("#" * 50)
        print(f"Now on {puzzle}")
        # check what we have
        puzzle_folder = os.path.join(dataset_folder, puzzle)
        general_files = os.listdir(puzzle_folder)
        solution_folders = [sol_fld for sol_fld in general_files if "solution" in sol_fld]
        print(f"Found {len(solution_folders)} solution folders")
        print(solution_folders)

        if len(solution_folders) > 0:
            with open(os.path.join(puzzle_folder, "ground_truth.json"), 'r') as gtj:
                ground_truth = json.load(gtj)
            with open(os.path.join(puzzle_folder, f"parameters_{puzzle}.json"), 'r') as gtj:
                img_parameters = json.load(gtj)
            with open(os.path.join(puzzle_folder, f"compatibility_parameters.json"), 'r') as gtj:
                cmp_parameters = json.load(gtj)
            with open(os.path.join(puzzle_folder, f"line_matching_parameters.json"), 'r') as gtj:
                lmp_parameters = json.load(gtj)

            for solution_folder in solution_folders:
                print("-" * 50)
                print(f"Evaluating: {solution_folder}")
                solution_folder_full_path = os.path.join(dataset_folder, puzzle, solution_folder)
                p_final_path = os.path.join(solution_folder_full_path, 'p_final.mat')
                p_final = loadmat(p_final_path)['p_final']
                anchor_id = np.squeeze(loadmat(p_final_path)['anchor']).item()
                anc_position = np.squeeze(loadmat(p_final_path)['anc_position'])

                num_pcs = img_parameters['num_pieces']
                gt_coord = np.zeros([num_pcs, 3])
                fin_solution = np.zeros([num_pcs, 3])
                for j in range(num_pcs):

                    # GT_'rotation' is angle to apply to roll-back to the original view
                    gt_coord[j, 2] = (+1) * ground_truth[f"piece_{j:04d}"]['rotation']
                    # GT saves XY-shift from original position to the center of image
                    # while the ref.point [0,0] is in up-left angle of image
                    gt_coord[j, 0:2] = (-1) * np.asarray(ground_truth[f"piece_{j:04d}"]['translation'][::-1])
                    fin_solution[j, :] = np.unravel_index(np.argmax(p_final[:, :, :, j]), p_final[:, :, :, j].shape)

                # Final solution in YXZ coordinates (anchor is in ref.point = [0,0,0])
                fin_sol = fin_solution - anc_position
                # Final solution in PX coordinate (anchor is in ref.point = [0,0,0])
                fin_sol_px = np.zeros([num_pcs, 3])
                fin_sol_px[:, 0:2] = fin_sol[:, 0:2] * cmp_parameters['xy_step']
                fin_sol_px[:, 2] = fin_sol[:, 2] * cmp_parameters['theta_step']

                # print(f"  ")
                # print(f"Fin Solution in YX :")
                # print(fin_sol)
                # print(f"  ")
                # print(f"Fin Solution in px:")
                # print(fin_sol_px)

                # anchor rotation in GT (safed)
                anc_coord = gt_coord[anchor_id, 0:2]
                anc_rot = gt_coord[anchor_id, 2]
                print(f"anchor ID: {anchor_id} rotation: {anc_rot}")
                print(f"--------------------------")

                # GT - shifted and rotated in PX (anchor is in ref.point = [0,0,0])
                gt_shift = np.zeros([num_pcs, 3])
                gt_shift[:, 0:2] = gt_coord[:, 0:2] - anc_coord
                gt_shift[:, 2] = gt_coord[:, 2] - anc_rot
                #print(f"GT non rotated in px :")
                #print(gt_shift)

                # 2. Translate to yxz space
                gt_yxz = np.zeros([num_pcs, 3])
                gt_yxz[:, 0:2] = np.round(gt_shift[:, 0:2]/cmp_parameters['xy_step'])
                gt_yxz[:, 2] = gt_shift[:, 2]/cmp_parameters['theta_step']
                #print(f"GT non rotated in YX :")
                #print(gt_yxz)

                # 3. Rotate GT according anchor rotation - CHECK!!!
                gt_rot = np.zeros([num_pcs, 3])
                if anc_rot == 0:
                    gt_rot[:, 0:2] = gt_yxz[:, 0:2]
                else:
                    if anc_rot == 90 or anc_rot == -270:
                        gt_rot[:, 0] = gt_yxz[:, 1]   # y_new = +x
                        gt_rot[:, 1] = -gt_yxz[:, 0]  # x_new = -y
                    elif anc_rot == -90 or anc_rot == 270:
                        gt_rot[:, 0] = -gt_yxz[:, 1]  # y_new = -x
                        gt_rot[:, 1] = gt_yxz[:, 0]   # x_new = +y
                    elif anc_rot == 180 or anc_rot == -180:
                        gt_rot[:, 0] = -gt_yxz[:, 0]  # y_new = -x
                        gt_rot[:, 1] = -gt_yxz[:, 1]  # x_new = -y
                gt_rot[:, 2] = gt_yxz[:, 2]
                print(f"  ")
                print(f"GT shifted and rotated in YX :")
                print(gt_rot)

                fin_sol = fin_solution - anc_position
                print(f"Fin Solution in YX :")
                print(fin_sol)

                print(f"--------------------------")
                gt_rot_px = np.zeros([num_pcs, 3])
                gt_rot_px[:, 0:2] = gt_rot[:, 0:2]*cmp_parameters['xy_step']
                gt_rot_px[:, 2] = gt_rot[:, 2]*cmp_parameters['theta_step']
                print(f"GT shifted and rotated in PX :")
                print(gt_rot_px)

                print(f"Fin Solution in PX :")
                print(fin_sol_px)

                print(">>> EVALUATION <<<")
                k_tol = 2     # tolerance shift 2 steps or more?
                errors_xy = np.sum(np.abs(gt_rot[:, 0:2]-fin_sol[:, 0:2]) <= k_tol, axis=1) == 2
                errors_px = np.sum(np.abs(gt_rot_px[:, 0:2]-fin_sol_px[:, 0:2]) <= (k_tol*cmp_parameters['xy_step']), axis=1) == 2
                errors_rot = (gt_rot[:,2] == fin_sol[:,2])

                Dir_accur_xy = np.round(np.mean(errors_xy),2)
                Dir_accur_px = np.round(np.mean(errors_px),2)
                Dir_accur_rot = np.round(np.mean(errors_rot),2)

                mean_xy_err = np.round(1- Dir_accur_xy, 2)
                mean_px_err = np.round(1 -Dir_accur_px, 2)
                mean_rot_err =np.round(1 -Dir_accur_rot, 2)

                evaluation_dict = {
                    'correct_in_PX_dist': Dir_accur_px,
                    'correct_on_rot': Dir_accur_rot,
                    'correct_on_xy': Dir_accur_xy,
                    'average_error_PX': mean_px_err,
                    'average_error_rot': mean_rot_err,
                    'average_error_xy': mean_xy_err,
                    'errors_px_list': errors_px.tolist(),
                    'errors_rot_list': errors_rot.tolist(),
                    'errors_yx_list': errors_xy.tolist(),
                }
                # for kk in evaluation_dict.keys():
                #      print(type(evaluation_dict[kk]), kk)
                with open(os.path.join(solution_folder_full_path, 'evaluation_2.json'), 'w') as ej:
                    json.dump(evaluation_dict, ej, indent=3)

                # check for p final
                # p_finals = [pfpath for pfpath in os.listdir(solution_folder_full_path) if "p_final" in pfpath]
                # print(f"found {len(p_finals)} p final matrices")
                # print(p_finals)
                # print(f"p final shape {p_final.shape}, anchor piece {anchor_id} in {anc_position}")

                print(f"Done with image {solution_folder} of {puzzle}")
        else:
            print("no solution found, skipping")

        print(f"Done with image {puzzle}")

    # pdb.set_trace()
    print("#" * 50)
    print("FINISHED")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('-d', '--dataset', type=str, default='synthetic_irregular_9_pieces_by_drawing_coloured_lines_28_02_2024', help='dataset folder')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')  # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-p', '--puzzle', type=str, default='', help='puzzle folder')
    args = parser.parse_args()

    main(args)
