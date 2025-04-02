import scipy 
import pdb 
import numpy as np 
import argparse 
import matplotlib.pyplot as plt 
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
        
        # output folder 
        puzzle_folder = os.path.join(dataset_folder, puzzle)
        output_folder = os.path.join(puzzle_folder, fnames.evaluation_folder_name)
        output_qualitative_folder = os.path.join(output_folder, 'qualitative_evaluation')
        output_quantitative_folder = os.path.join(output_folder, 'quantitative_evaluation')
        output_visualization_folder = os.path.join(output_folder, 'visualization_evaluation')
        os.makedirs(output_qualitative_folder, exist_ok=True)
        os.makedirs(output_quantitative_folder, exist_ok=True)
        os.makedirs(output_visualization_folder, exist_ok=True)

        
        solution_folder_names = [solution_f for solution_f in os.listdir(puzzle_folder) if 'solution_anchor' in solution_f]
        if len(solution_folder_names) > 0:
            if args.all_anchors is False:

                given_anchor = args.anchor
                if given_anchor < 0:
                    # take the first! (there should be only one)
                    solutions_anchors_folders = [solution_folder_names[0]]
                else:
                    print('not implemented yet, we are changing the names too much! let the script find the anchor or use `-aa` to evaluate all!')

            else: # all folders
                solutions_anchors_folders = solution_folder_names 
                #[solution_f for solution_f in os.listdir(root_path) if 'solution_anchor' in solution_f]
            with open(os.path.join(puzzle_folder, "ground_truth.json"), 'r') as gtj:
                ground_truth = json.load(gtj)
            with open(os.path.join(puzzle_folder, f"parameters_{puzzle}.json"), 'r') as gtj:
                img_parameters = json.load(gtj)
            with open(os.path.join(puzzle_folder, f"compatibility_parameters.json"), 'r') as gtj:
                cmp_parameters = json.load(gtj)
            with open(os.path.join(puzzle_folder, f"line_matching_parameters.json"), 'r') as gtj:
                lmp_parameters = json.load(gtj)

            for solution_folder_anc in solutions_anchors_folders:
                cost_name = solution_folder_anc.split('_')[-2] 
                # read p matrix
                solution_path = os.path.join(puzzle_folder, solution_folder_anc, 'p_final.mat')
                solution_matrix = scipy.io.loadmat(solution_path)
                anchor_idx = solution_matrix['anchor'] 
                # anchor = ((np.ceil(cfg.num_patches_side/2) - 1)*(cfg.num_patches_side+1)).astype(int) #solution_matrix['anchor']
                anchor_idx = np.squeeze(anchor_idx).item()
                anchor_pos = solution_matrix['anc_position']
                anchor_pos = np.squeeze(anchor_pos)
                p_final = solution_matrix['p_final']

                # visual solution
                num_pieces = args.num_pieces
                offset_start = get_offset(anchor_idx, anchor_pos, num_pieces)
                #anchor_pos[:2] - anchor_pos_in_puzzle
                pieces_folder = os.path.join(puzzle_folder, f"{fnames.pieces_folder}")
                squared_solution_img = get_visual_solution_from_p(p_final, pieces_folder, img_parameters['piece_size'], offset_start, num_pieces, cmp_parameters, crop=True)
                
                # ref image
                im_ref = plt.imread(os.path.join(puzzle_folder, fnames.ref_image, f"output_{puzzle[:11]}.png"))

                ### EVALUATION 
                # simple evaluation (# of pieces in correct position)
                num_correct_pieces, visual_correct = simple_evaluation(p_final, num_pieces, offset_start, anchor_idx, verbosity=args.verbosity)
                perc_correct = num_correct_pieces / (num_pieces**2)

                # vector evaluation
                true_absolute_solution = get_true_solution_vector(num_pieces)
                pred_solutions = get_pred_solution_vector(p_final, num_pieces)
                num_correct_pcs_vector = simple_evaluation_vector(pred_solutions, true_absolute_solution, anchor_pos=anchor_pos, anchor_idx=anchor_idx, num_pieces=num_pieces)
                perc_correct_vector = num_correct_pcs_vector / (num_pieces**2)

                # MSE error (pixel-wise difference)
                measure = 'mse'
                if np.min(squared_solution_img.shape[:2]) > 0:
                    MSError = pixel_difference(squared_solution_img, im_ref, measure=measure)
                    solved_img_output_path = os.path.join(output_qualitative_folder, f'evaluated_solution_anchor{anchor_idx}_bbg_{cost_name}.jpg')
                    plt.imsave(solved_img_output_path, np.clip(squared_solution_img, 0, 1))
                    squared_solution_img[squared_solution_img == 0] = 1
                    MSError_white = pixel_difference(squared_solution_img, im_ref, measure=measure)
                else:
                    MSError = 1
                    MSError_white = 1
                # neighbours comparison
                neighbours_val = neighbor_comparison(get_sol_from_p(p_final=p_final), num_pieces, offset_start)

                eval_res = {
                    'correct': perc_correct,
                    'correct_vector': num_correct_pcs_vector.tolist(),
                    'neighbours': neighbours_val,
                    'pixel_b': MSError,
                    'pixel_w': MSError_white
                }
                json_output_path = os.path.join(output_quantitative_folder, f'evaluation_anchor{anchor_idx}_{cost_name}.json')
                with open(json_output_path, 'w') as jf: 
                    json.dump(eval_res, jf, indent=2)

                print("\nEVALUATION")
                print(f"Correct: {perc_correct * 100:.03f} %")
                print(f"Correct Vector: {perc_correct_vector * 100:.03f} %")
                print(f"Neighbours: {neighbours_val * 100:.03f} %")
                print(f"Pixel-wise B({measure}): {MSError}")
                print(f"Pixel-wise W({measure}): {MSError_white}\n")

                if np.min(squared_solution_img.shape[:2]) > 0:
                    solved_img_output_path = os.path.join(output_qualitative_folder, f'evaluated_solution_anchor{anchor_idx}_wbg_{cost_name}.jpg')
                    plt.imsave(solved_img_output_path, np.clip(squared_solution_img, 0, 1))
                    #cv2.imwrite(solved_img_output_path, cv2.cvtColor((squared_solution_img*255).astype(np.uint8), cv2.COLOR_BAYER_BG2BGR))

                anc_xy_pos = get_xy_position(anchor_idx, num_pieces, offset_start=0)
                plt.figure(figsize=(32,32))
                plt.suptitle(f"{args.puzzle}\nanchor = {anchor_idx} ( position {anc_xy_pos})\ncorrect pieces = {num_correct_pieces / (num_pieces**2) * 100:.03f}%\nneighbours = {neighbours_val * 100:.03f}%\nMSE = {MSError:.03f}", fontsize=52)
                plt.subplot(221)
                plt.title('reference image', fontsize=32)
                plt.imshow(im_ref, cmap='gray')
                if np.min(squared_solution_img.shape[:2]) > 0:
                    plt.subplot(223)
                    plt.title(f'solution of the puzzle ({num_correct_pieces / (num_pieces**2) * 100:.03f} %)', fontsize=32)
                    plt.imshow(squared_solution_img)
                plt.subplot(222)
                plt.title(f'correct pieces: {num_correct_pieces}\nanchor: blue\ncorrectly solved: green\nwrong: red', fontsize=32)
                plt.imshow(visual_correct)
                if np.min(squared_solution_img.shape[:2]) > 0:
                    plt.subplot(224)
                    plt.title(f'showing the grid of pieces', fontsize=32)
                    plt.imshow(squared_solution_img)
                    for j in range(1, num_pieces):
                        plt.axline([0, j*cmp_parameters['piece_size']], slope=0, color='blue')
                        plt.axline([j*cmp_parameters['piece_size'], 0], slope=np.inf, color='blue')

                anc_center = (anc_xy_pos) * cmp_parameters['piece_size']
                box_anchor = np.asarray([[anc_center[1], anc_center[0]], [anc_center[1], anc_center[0]+cmp_parameters['piece_size']], \
                                [anc_center[1]+cmp_parameters['piece_size'], anc_center[0]+cmp_parameters['piece_size']], [anc_center[1]+cmp_parameters['piece_size'], anc_center[0]], [anc_center[1], anc_center[0]]])
                plt.plot(box_anchor[:,1], box_anchor[:,0], color='green', linewidth=5)
                if args.visualize is True:
                    plt.show()
                else:
                    outputpath = os.path.join(output_visualization_folder, f'visualization_solution_anchor{anchor_idx}_{cost_name}.png')
                    plt.tight_layout()
                    plt.savefig(outputpath)
                print(f'done with anchor {anchor_idx}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('-d', '--dataset', type=str, default='manual_lines', help='dataset folder')                     # repair, wikiart, manual_lines, architecture
    parser.add_argument('-p', '--puzzle', type=str, default='', help='puzzle folder')                             # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-n', '--num_pieces', type=int, default=8, help='number of pieces (per side)')                  # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-a', '--anchor', type=int, default=-1, help='anchor piece (index)')                            # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-v', '--visualize', default=False, action='store_true', help='use to show the solution')       # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-aa', '--all_anchors', default=False, action='store_true', help='use to evaluate all anchors of this puzzle')       # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('--verbosity', type=int, default=1, help='level of logging/printing (0 --> nothing, higher --> more printed stuff)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    
    args = parser.parse_args()

    main(args)
