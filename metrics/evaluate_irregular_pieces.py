import scipy 
from scipy.io import loadmat
import pdb 
import numpy as np 
import argparse 
import matplotlib.pyplot as plt 
from configs import folder_names as fnames
import os 
import cv2
from metrics.metrics_utils import include_rotation
import json 
import pandas as pd

def main(args):

    dataset_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset)
    print(dataset_folder)
    if args.puzzle == '':  
        puzzles = os.listdir(dataset_folder)
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(dataset_folder, puz)) is True]
    else:
        puzzles = [args.puzzle]

    tot_puz_names = []
    tot_sol_names = []
    tot_correct = []
    tot_correct_xy = []
    tot_correct_rot = []
    tot_neigh = []
    tot_neigh_xy = []
    tot_neigh_rot = []

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
            with open(os.path.join(puzzle_folder, "neighbours.json"), 'r') as gtj:
                neighbours = json.load(gtj)
            with open(os.path.join(puzzle_folder, f"parameters_{puzzle}.json"), 'r') as gtj:
                img_parameters = json.load(gtj)
            with open(os.path.join(puzzle_folder, "compatibility_parameters.json"), 'r') as gtj:
                cmp_parameters = json.load(gtj)
            with open(os.path.join(puzzle_folder, "line_matching_parameters.json"), 'r') as gtj:
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

                # ground truth 
                # we saved them as negative (both translation and rotation)
                gt_xy_anc = -1 * np.asarray(ground_truth[f"piece_{anchor_id:04d}"]['translation'][::-1])
                gt_rotation_anc = -1 * ground_truth[f"piece_{anchor_id:04d}"]['rotation']
                print(f"GT xy anc is: {gt_xy_anc}")
                print(f"GT rot anc is: {gt_rotation_anc}")
                # we place the ancor in the center of the canvas
                shift_gt_anc2canvas = gt_xy_anc - center_canvas
                # if gt rot is not 0, we need to shift all the coordinates
                print(f"aligning to canvas --> shift_gt_anc2canvas: {shift_gt_anc2canvas}")
                print()

                # now we can get the position of the anchor                 
                anc_position_xy = anc_position[:2] * cmp_parameters['xy_step']
                # and the shift to get it to the center of the canvas
                shift_anc2canvas = anc_position_xy - center_canvas
                anc_position_on_canvas = center_canvas
                print(f"\nAnchor {anchor_id} in {anc_position_on_canvas} on a canvas of {canvas.shape}\n")
                print(f"anchor was: {anc_position_xy} rotated of {0}")
                print(f"aligning to canvas on xy --> shift_anc2canvas: {shift_anc2canvas}")
                
                # rotation
                # we need to have the rotation delta between the ground truth and the anchor
                # everything will be solved as the "original" image, meaning:
                #  the anchor is selected with a certain rotation (usually 0), but in the GT it has another rotation
                #  we rotate all the estimated to align the image to the original one!
                anc_rotation = anc_position[2] * cmp_parameters['theta_step']
                rot_diff = gt_rotation_anc - anc_rotation

                ######
                # NOW:
                # - we have the anchor gt in the center of the canvas (and the shift to align the other pieces)
                # - we have the anchor of estimation in the cneter (and its shift)
                # - we have the rotation delta, which means, all the estimated (trans and rot) needs to be rotated by that
                #
                # therefore we need to:
                # - for each gt piece, apply 2 shifts: 1 to align to anchor gt and 1 to align to center canvas
                # - for each est piece, transform its position by rotating him and its shift to align to the center of canvas!
                #       rotating the shift means changing the xy coords (example 180 deg invert them xy = - xy)
                ######              
                num_pcs = img_parameters['num_pieces']
                errors_xy = np.zeros(num_pcs)
                errors_rot = np.zeros(num_pcs)
                correct_xy = np.zeros(num_pcs)
                correct_rot = np.zeros(num_pcs)
                direct_correct = np.zeros(num_pcs)
                neigh_xy = np.zeros(num_pcs)
                neigh_rot = np.zeros(num_pcs)
                neighbours_correct = np.zeros(num_pcs)
                neighbours_nums = np.zeros(num_pcs)

                est_pos = np.zeros((num_pcs, 3))
                for p in range(num_pcs):
                    est_pos[p, :] = np.unravel_index(np.argmax(p_final[:,:,:,p]), p_final[:,:,:,p].shape)

                for j in range(num_pcs):

                    gt_rot_piece = ground_truth[f"piece_{j:04d}"]['rotation']
                    # here we calculate shift between ground truth absolute and with respect to the anchor!
                    gt_xy = -1 * np.asarray(ground_truth[f"piece_{j:04d}"]['translation'][::-1])
                    gt_anc = gt_xy + gt_xy_anc
                    gt_xy_canvas = gt_xy - shift_gt_anc2canvas
                    
                    # here the solution (from the solver)
                    solution_piece = est_pos[j, :] #np.unravel_index(np.argmax(p_final[:,:,:,j]), p_final[:,:,:,j].shape)
                    est_xy = np.asarray(solution_piece[:2]) * cmp_parameters['xy_step']
                    est_rot = solution_piece[2] * cmp_parameters['theta_step']
                    est_xy_canvas = est_xy - include_rotation(shift_anc2canvas, rot_diff)

                    error_xy = np.sqrt(np.sum(np.square(gt_xy_canvas - est_xy_canvas)))
                    error_rot = np.sqrt(np.square(np.abs(gt_rot_piece - est_rot))) % 360
                    if error_xy < cmp_parameters['xy_step']: #np.isclose(error_xy, 0):
                        correct_xy[j] = 1
                    if error_rot < cmp_parameters['theta_step'] or np.isclose(error_rot, 360): #np.isclose(error_rot, 0):
                        correct_rot[j] = 1
                    if correct_xy[j] == 1 and correct_rot[j] == 1:
                        direct_correct[j] = 1
                    errors_xy[j] = error_xy
                    errors_rot[j] = error_rot

                    # neighbours 
                    neighs_j = neighbours['numbers'][f"{j:d}"]
                    neighbours_nums[j] = len(neighs_j)
                    for neigh_j in neighs_j: # neigh_j is a number !
                        rel_xy_gt = - np.asarray(ground_truth[f"piece_{j:04d}"]['translation'][::-1]) + np.asarray(ground_truth[f"piece_{neigh_j:04d}"]['translation'][::-1])
                        rel_rot_gt =  np.asarray(ground_truth[f"piece_{j:04d}"]['rotation']) - np.asarray(ground_truth[f"piece_{neigh_j:04d}"]['rotation'])
                        rel_xy_est = (est_pos[j, :2] - est_pos[neigh_j, :2]) * cmp_parameters['xy_step']
                        rel_rot_est = (est_pos[j, 2] - est_pos[neigh_j, 2]) * cmp_parameters['theta_step']
                        nxy = False
                        nr = False
                        if (np.abs(rel_xy_gt[0] - rel_xy_est[0]) < cmp_parameters['xy_step']) and (np.abs(rel_xy_gt[1] - rel_xy_est[1]) < cmp_parameters['xy_step']):
                            neigh_xy[j] += 1
                            nxy = True
                        if rel_rot_gt < 0: 
                            rel_rot_gt += 360
                        if rel_rot_est < 0: 
                            rel_rot_est += 360
                        if (np.abs(rel_rot_gt - rel_rot_est) < cmp_parameters['theta_step']) or np.isclose(rel_rot_gt, rel_rot_est):
                            neigh_rot[j] += 1
                            nr = True
                        if nxy == True and nr == True:
                            neighbours_correct[j] += 1
                    
                    print("-" * 40)
                    print(f"piece {j}")
                    print(">>> ESTIMATED  <<<")
                    print(f"p: {solution_piece}\nest_xy: {est_xy}\nest_xy_canvas: {est_xy_canvas}")
                    print(f"est_rot: {est_rot}")
                    print(">>> GT         <<<")
                    print(f"gt_xy: {gt_xy}\ngt_anc: {gt_anc}\ngt_xy_canvas: {gt_xy_canvas}")
                    print(f"gt_rot_piece: {gt_rot_piece}")
                    print(">>> EVALUATION <<<")
                    print(f"piece {j} is placed at ({est_xy_canvas}, {solution_piece[2]}), gt is at ({gt_xy_canvas}, {gt_rot_piece})")
                    print(f"error is: {error_xy} on xy, {error_rot} on rotation")
                    print(">>> NEIGHBOURS <<<")
                    print(f"piece {j} has {neighbours_nums[j]} neighbours:\n{neighbours_correct[j]} were correct ({neigh_xy[j]} on xy and {neigh_rot[j]} on rotation)")

                mean_xy_err = np.mean(errors_xy)
                mean_rot_err = np.mean(errors_rot)
                num_correct_pcs_xy = np.sum(correct_xy)
                num_correct_pcs_rot = np.sum(correct_rot)
                num_correct_pcs = np.sum(direct_correct)
                print("\n")
                print("*" * 50)
                print("CORRECT")
                print(f"XY: {num_correct_pcs_xy} / {num_pcs} : {num_correct_pcs_xy/num_pcs:.02f}")
                print(f"ROT: {num_correct_pcs_rot} / {num_pcs} : {num_correct_pcs_rot/num_pcs:.02f}")
                print(f"Both: {num_correct_pcs}  / {num_pcs} : {num_correct_pcs/num_pcs:.02f}")
                print("AVERAGE DISTANCE")
                print(f"XY: {mean_xy_err:.3f}")
                print(f"ROT: {mean_rot_err:.3f}")
                perc_correct_neighbours = neighbours_correct / neighbours_nums
                perc_correct_neighbours_xy = neigh_xy / neighbours_nums
                perc_correct_neighbours_rot = neigh_rot / neighbours_nums
                mean_correct_neigh = np.mean(perc_correct_neighbours)
                mean_correct_neigh_xy = np.mean(perc_correct_neighbours_xy)
                mean_correct_neigh_rot = np.mean(perc_correct_neighbours_rot)
                print("*" * 50)
                print("NEIGHBOURS")
                print(f"avg ratio: {mean_correct_neigh:.03f}")
                print(f"avg ratio xy: {mean_correct_neigh_xy:.03f}")
                print(f"avg ratio rot: {mean_correct_neigh_rot:.03f}")
                print("*" * 50)

                evaluation_dict = {
                    'correct': num_correct_pcs,
                    'correct_on_xy': num_correct_pcs_xy,
                    'correct_on_rot': num_correct_pcs_rot,
                    'average_dist_xy': mean_xy_err,
                    'average_dist_rot': mean_rot_err,
                    'average_neighbours': mean_correct_neigh,
                    'average_neighbours_xy': mean_correct_neigh_xy,
                    'average_neighbours_rot': mean_correct_neigh_rot,
                    'neighbours_sum_list': neighbours_nums.tolist(),
                    'average_neighbours_num': np.mean(neighbours_nums),
                    'errors_xy_list': errors_xy.tolist(),
                    'errors_rot_list': errors_rot.tolist(),
                }
                # for kk in evaluation_dict.keys():
                #     print(type(evaluation_dict[kk]), kk)
                with open(os.path.join(solution_folder_full_path, 'evaluation.json'), 'w') as ej:
                    json.dump(evaluation_dict, ej, indent=3)
                

                tot_puz_names.append(puzzle)
                tot_sol_names.append(solution_folder)
                tot_correct.append(num_correct_pcs/num_pcs)
                tot_correct_xy.append(num_correct_pcs_xy/num_pcs)
                tot_correct_rot.append(num_correct_pcs_rot/num_pcs)
                tot_neigh.append(mean_correct_neigh)
                tot_neigh_xy.append(mean_correct_neigh_xy)
                tot_neigh_rot.append(mean_correct_neigh_rot)
                #print(estimated_pos_piece)

                # check for p final 
                # p_finals = [pfpath for pfpath in os.listdir(solution_folder_full_path) if "p_final" in pfpath]
                # print(f"found {len(p_finals)} p final matrices")
                # print(p_finals)
                #print(f"p final shape {p_final.shape}, anchor piece {anchor_id} in {anc_position}") 
        else:
            print("no solution found, skipping")

    metrics_df = pd.DataFrame()
    metrics_df['puzzle_names'] = tot_puz_names
    metrics_df['solution_names'] = tot_sol_names
    metrics_df['direct'] = tot_correct
    metrics_df['neighbours'] = tot_neigh
    metrics_df['direct_xy'] = tot_correct_xy
    metrics_df['direct_rot'] = tot_correct_rot
    metrics_df['neighbours_xy'] = tot_neigh_xy
    metrics_df['neighbours_rot'] = tot_neigh_rot

    metrics_df.to_csv(os.path.join(dataset_folder, 'metrics.csv'))
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
