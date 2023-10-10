import scipy 
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
    pixel_difference, neighbor_comparison
import json 

def main(args):

    root_path = os.path.join(f"{fnames.output_dir}_{cfg.num_patches_side}x{cfg.num_patches_side}", args.dataset, args.puzzle)
    
    # read p matrix
    solution_path = os.path.join(root_path, fnames.solution_folder_name, 'p_final.mat')
    solution_matrix = scipy.io.loadmat(solution_path)
    anchor = ((np.ceil(cfg.num_patches_side/2) - 1)*(cfg.num_patches_side+1)).astype(int) #solution_matrix['anchor'] 
    anchor = np.squeeze(anchor).item()
    anchor_pos = solution_matrix['anc_position']
    anchor_pos = np.squeeze(anchor_pos)
    p_final = solution_matrix['p_final']

    # visual solution
    num_pieces = args.num_pieces
    anchor_pos_in_puzzle = np.asarray([anchor % num_pieces, anchor // num_pieces])
    offset_start = anchor_pos[:2] - anchor_pos_in_puzzle
    pieces_folder = os.path.join(root_path, f"{fnames.pieces_folder}")
    squared_solution_img = get_visual_solution_from_p(p_final, pieces_folder, cfg.piece_size, offset_start, num_pieces)

    # ref image
    im_ref = plt.imread(os.path.join(fnames.data_path, args.dataset, fnames.images_folder, f"{args.puzzle}.jpg"))

    ### EVALUATION 
    # simple evaluation (# of pieces in correct position)
    num_correct_pieces, visual_correct = simple_evaluation(p_final, num_pieces, offset_start)
    perc_correct = num_correct_pieces / (num_pieces**2)
    # MSE error (pixel-wise difference)
    measure = 'rmse'
    MSError = pixel_difference(squared_solution_img, im_ref, measure=measure)
    # neighbours comparison
    neighbours_val = neighbor_comparison(get_sol_from_p(p_final=p_final), num_pieces, offset_start)

    
    # output folder 
    output_folder = os.path.join(root_path, fnames.evaluation_folder_name)

    eval_res = {
        'correct': perc_correct,
        'neighbours': neighbours_val,
        'pixel': MSError
    }
    json_output_path = os.path.join(output_folder, 'evaluation.json')
    with open(json_output_path, 'w') as jf: 
        json.dump(eval_res, jf, indent=2)

    print("\nEVALUATION")
    print(f"Correct: {perc_correct * 100:.03f} %")
    print(f"Neighbours: {neighbours_val * 100:.03f} %")
    print(f"Pixel-wise ({measure}): {MSError}\n")

    solved_img_output_path = os.path.join(output_folder, 'solved.jpg')
    cv2.imwrite(solved_img_output_path, squared_solution_img*255)

    plt.figure(figsize=(32,32))
    plt.suptitle(f"{args.puzzle}\ncorrect pieces = {num_correct_pieces / (num_pieces**2) * 100:.03f}%\nneighbours = {neighbours_val * 100:.03f}%\nMSE = {MSError:.03f}", fontsize=52)
    plt.subplot(131)
    plt.title('reference image', fontsize=32)
    plt.imshow(im_ref, cmap='gray')
    plt.subplot(133)
    plt.title(f'solution of the puzzle ({num_correct_pieces / (num_pieces**2) * 100:.03f} %)', fontsize=32)
    plt.imshow(squared_solution_img)
    plt.subplot(132)
    plt.title(f'correct pieces (yellow): {num_correct_pieces}', fontsize=32)
    plt.imshow(visual_correct)
    if args.visualize is True:
        plt.show()
    else:
        outputpath = os.path.join(output_folder, 'visualization_solution.png')
        plt.tight_layout()
        plt.savefig(outputpath)

    return 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('-d', '--dataset', type=str, default='manual_lines', help='dataset folder')   # repair, wikiart, manual_lines, architecture
    parser.add_argument('-p', '--puzzle', type=str, default='lines1', help='puzzle folder')           # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-n', '--num_pieces', type=int, default=8, help='number of pieces (per side)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-v', '--visualize', default=False, action='store_true', help='use to show the solution')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    args = parser.parse_args()

    main(args)