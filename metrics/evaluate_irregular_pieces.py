import scipy 
from scipy.io import loadmat
import pdb 
import numpy as np 
import argparse 
import matplotlib.pyplot as plt 
from configs import unified_cfg as cfg
from configs import folder_names as fnames
import os 
from solver.solverRotPuzzArgs import reconstruct_puzzle
import cv2
from metrics.metrics_utils import get_sol_from_p, get_visual_solution_from_p, simple_evaluation, \
    pixel_difference, neighbor_comparison, get_offset, get_true_solution_vector, \
        get_pred_solution_vector, get_xy_position, simple_evaluation_vector
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
            
            canvas = np.zeros((cmp_parameters['canvas_size'], cmp_parameters['canvas_size'], 3))
            center_canvas = np.asarray(canvas.shape[:2]) // 2

            
            for solution_folder in solution_folders:
                solution_folder_full_path = os.path.join(dataset_folder, puzzle, solution_folder)
                p_final_path = os.path.join(solution_folder_full_path, 'p_final.mat')
                p_final = loadmat(p_final_path)['p_final']
                anchor_id = np.squeeze(loadmat(p_final_path)['anchor']).item()
                anc_position = np.squeeze(loadmat(p_final_path)['anc_position'])
                shift_anc2canvas = anc_position[:2] - center_canvas
                print(f"\nAnchor {anchor_id} in {anc_position}, (shift: {shift_anc2canvas})")

                num_pcs = img_parameters['num_pieces']
                errors_xy = np.zeros(num_pcs)
                errors_rot = np.zeros(num_pcs)
                correct_xy = np.zeros(num_pcs)
                correct_rot = np.zeros(num_pcs)
                for j in range(num_pcs):
                    # here we calculate shift between ground truth absolute and with respect to the anchor!
                    gt_xy_orig = ground_truth[f"piece_{j:04d}"]['translation']
                    gt_xy = anc_position[:2] - shift_anc2canvas
                    gt_shift = gt_xy_orig - gt_xy
                    gt_rot_orig = ground_truth[f"piece_{j:04d}"]['rotation']
                    gt_rot = anc_position[2]
                    gt_rot_shift = gt_rot_orig - gt_rot
                    # here the solution (from the solver)
                    solution_piece = np.unravel_index(np.argmax(p_final[:,:,:,j]), p_final[:,:,:,j].shape)
                    est_xy = solution_piece[:2] - shift_anc2canvas
                    est_rot = (solution_piece[:2] + gt_rot_shift) % cmp_parameters['theta_grid_points']
                    error_xy = np.sqrt(np.sum(np.square(gt_xy - est_xy)))
                    error_rot = np.sqrt(np.square(np.abs(gt_rot - solution_piece[2])))
                    if np.isclose(error_xy, 0):
                        correct_xy[j] = 1
                    if np.isclose(error_rot, 0):
                        correct_rot[j] = 1
                    errors_xy[j] = error_xy
                    errors_rot[j] = error_rot
                    print("-" * 40)
                    print(f"piece {j} is placed at ({est_xy}, {solution_piece[2]}), gt was at ({gt_xy}, {gt_rot})")
                    print(f"error is: {error_xy} on xy, {error_rot} on rotation")

                mean_xy_err = np.mean(errors_xy)
                mean_rot_err = np.mean(errors_rot)
                num_correct_pcs_xy = np.sum(correct_xy)
                num_correct_pcs_rot = np.sum(correct_rot)
                print("*" * 50)
                print("CORRECT")
                print(f"XY: {num_correct_pcs_xy}")
                print(f"ROT: {num_correct_pcs_rot}")
                print("AVERAGE DISTANCE")
                print(f"XY: {mean_xy_err:.3f}")
                print(f"ROT: {mean_rot_err:.3f}")

                evaluation_dict = {
                    'correct_on_xy': num_correct_pcs_xy,
                    'correct_on_rot': num_correct_pcs_rot,
                    'average_dist_xy': mean_xy_err,
                    'average_dist_rot': mean_rot_err,
                    'errors_xy_list': errors_xy.tolist(),
                    'errors_rot_list': errors_rot.tolist(),
                }
                # for kk in evaluation_dict.keys():
                #     print(type(evaluation_dict[kk]), kk)
                with open(os.path.join(solution_folder_full_path, 'evaluation.json'), 'w') as ej:
                    json.dump(evaluation_dict, ej, indent=3)
                
                #print(estimated_pos_piece)

                # check for p final 
                # p_finals = [pfpath for pfpath in os.listdir(solution_folder_full_path) if "p_final" in pfpath]
                # print(f"found {len(p_finals)} p final matrices")
                # print(p_finals)
                #print(f"p final shape {p_final.shape}, anchor piece {anchor_id} in {anc_position}") 
        else:
            print("no solution found, skipping")
    #pdb.set_trace()
    print("#" * 50)
    print("FINISHED")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('-d', '--dataset', type=str, default='manual_lines', help='dataset folder')   
    parser.add_argument('--verbosity', type=int, default=1, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-p', '--puzzle', type=str, default='', help='puzzle folder')    
    

    # parser.add_argument('-n', '--num_pieces', type=int, default=8, help='number of pieces (per side)')                  # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    # parser.add_argument('-a', '--anchor', type=int, default=-1, help='anchor piece (index)')                            # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    # parser.add_argument('-v', '--visualize', default=False, action='store_true', help='use to show the solution')       # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    # parser.add_argument('-aa', '--all_anchors', default=False, action='store_true', help='use to evaluate all anchors of this puzzle')       # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    
    args = parser.parse_args()

    main(args)
