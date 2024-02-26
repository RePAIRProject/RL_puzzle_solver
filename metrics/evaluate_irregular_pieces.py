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
        get_pred_solution_vector, get_xy_position, simple_evaluation_vector, include_rotation
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
            
            canvas_size = np.round(np.sqrt(img_parameters['num_pieces']) * img_parameters['piece_size']).astype(int)
            #canvas_size = cmp_parameters['canvas_size']
            canvas = np.zeros((canvas_size, canvas_size, 3))
            center_canvas = np.asarray(canvas.shape[:2]) // 2

            
            for solution_folder in solution_folders:
                print("-" * 50)
                print(f"Evaluating: {solution_folder}")
                solution_folder_full_path = os.path.join(dataset_folder, puzzle, solution_folder)
                p_final_path = os.path.join(solution_folder_full_path, 'p_final.mat')
                p_final = loadmat(p_final_path)['p_final']
                anchor_id = np.squeeze(loadmat(p_final_path)['anchor']).item()
                anc_position = np.squeeze(loadmat(p_final_path)['anc_position'])
                anc_position_xy = anc_position[:2] * cmp_parameters['xy_step']
                shift_anc2canvas = anc_position_xy - center_canvas
                anc_position_on_canvas = center_canvas
                anc_rotation = anc_position[2] * cmp_parameters['theta_step']
                print(f"\nAnchor {anchor_id} in {anc_position_on_canvas} on a canvas of {canvas.shape}\n")
                print(f"anchor was: {anc_position_xy} rotated of {anc_rotation}")
                print(f"aligning to canvas on xy --> shift_anc2canvas: {shift_anc2canvas}")
                
                gt_xy_anc = -1 * np.asarray(ground_truth[f"piece_{anchor_id:04d}"]['translation'][::-1])
                gt_rotation_anc = -1 * ground_truth[f"piece_{anchor_id:04d}"]['rotation']
                shift_rot_gt_anc2canvas = gt_rotation_anc - anc_rotation
                print(f"GT xy anc was: {gt_xy_anc}")
                shift_gt_anc2canvas = gt_xy_anc - center_canvas
                # if gt rot is not 0, we need to shift all the coordinates
                print(f"aligning to canvas --> shift_gt_anc2canvas: {shift_gt_anc2canvas}")
                print(f"aligning to canvas on rot --> shift_rot_gt_anc2canvas: {shift_rot_gt_anc2canvas}")
                print()
                num_pcs = img_parameters['num_pieces']
                errors_xy = np.zeros(num_pcs)
                errors_rot = np.zeros(num_pcs)
                correct_xy = np.zeros(num_pcs)
                correct_rot = np.zeros(num_pcs)
                correct = np.zeros(num_pcs)
                for j in range(num_pcs):
                    gt_rot_orig = -1 * ground_truth[f"piece_{j:04d}"]['rotation']
                    gt_rot = gt_rot_orig + shift_rot_gt_anc2canvas
                    # here we calculate shift between ground truth absolute and with respect to the anchor!
                    gt_xy = -1 * np.asarray(ground_truth[f"piece_{j:04d}"]['translation'][::-1])
                    gt_xy_canvas = gt_xy - shift_gt_anc2canvas
                    gt_xy_canvas_rot = include_rotation(gt_xy_canvas, shift_rot_gt_anc2canvas)
                    
                    # here the solution (from the solver)
                    solution_piece = np.unravel_index(np.argmax(p_final[:,:,:,j]), p_final[:,:,:,j].shape)
                    est_xy = np.asarray(solution_piece[:2]) * cmp_parameters['xy_step']
                    est_xy_canvas = est_xy - shift_anc2canvas
                    est_rot = solution_piece[2] * cmp_parameters['theta_step']

                    error_xy = np.sqrt(np.sum(np.square(gt_xy_canvas - est_xy_canvas)))
                    error_rot = np.sqrt(np.square(np.abs(gt_rot - est_rot)))
                    if error_xy < cmp_parameters['xy_step']: #np.isclose(error_xy, 0):
                        correct_xy[j] = 1
                    if error_rot < cmp_parameters['theta_step'] or np.isclose(error_rot, 360): #np.isclose(error_rot, 0):
                        correct_rot[j] = 1
                    if correct_xy[j] == 1 and correct_rot[j] == 1:
                        correct[j] = 1
                    errors_xy[j] = error_xy
                    errors_rot[j] = error_rot
                    print("-" * 40)
                    print(f"piece {j}")
                    print(">>> ESTIMATED  <<<")
                    print(f"p: {solution_piece}\nest_xy: {est_xy}\nest_xy_canvas: {est_xy_canvas}")
                    print(f"est_rot: {est_rot}")
                    print(">>> GT         <<<")
                    print(f"gt_xy: {gt_xy}\ngt_xy_canvas: {gt_xy_canvas}")
                    print(f"gt_rot_orig: {gt_rot_orig}\ngt_rot: {gt_rot}")
                    print(">>> EVALUATION <<<")
                    print(f"piece {j} is placed at ({est_xy_canvas}, {solution_piece[2]}), gt was at ({gt_xy_canvas}, {gt_rot})")
                    print(f"error is: {error_xy} on xy, {error_rot} on rotation")

                mean_xy_err = np.mean(errors_xy)
                mean_rot_err = np.mean(errors_rot)
                num_correct_pcs_xy = np.sum(correct_xy)
                num_correct_pcs_rot = np.sum(correct_rot)
                num_correct_pcs = np.sum(correct)
                print("*" * 50)
                print("CORRECT")
                print(f"XY: {num_correct_pcs_xy}")
                print(f"ROT: {num_correct_pcs_rot}")
                print(f"Both: {num_correct_pcs}")
                print("AVERAGE DISTANCE")
                print(f"XY: {mean_xy_err:.3f}")
                print(f"ROT: {mean_rot_err:.3f}")

                evaluation_dict = {
                    'correct': num_correct_pcs,
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
