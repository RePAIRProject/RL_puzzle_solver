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

def get_sol_from_p(p_final):
    noPatches = p_final.shape[3]
    I = np.zeros((noPatches, 1))
    m = np.zeros((noPatches, 1))
    for j in range(noPatches):
        p_tmp_final = p_final[:, :, :, j]
        m[j, 0], I[j, 0] = np.max(p_tmp_final), np.argmax(p_tmp_final)

    I = I.astype(int)
    i1, i2, i3 = np.unravel_index(I, p_tmp_final.shape)

    fin_sol = np.concatenate((i1, i2, i3), axis=1)
    return fin_sol 

def main(args):

    # read p matrix
    solution_matrix = scipy.io.loadmat(args.solution)
    anchor = ((np.ceil(cfg.num_patches_side/2) - 1)*(cfg.num_patches_side+1)).astype(int) #solution_matrix['anchor'] 
    anchor = np.squeeze(anchor).item()
    anchor_pos = solution_matrix['anc_position']
    anchor_pos = np.squeeze(anchor_pos)
    p_final = solution_matrix['p_final']

    num_pieces = args.pieces

    anchor_pos_in_puzzle = np.asarray([anchor % num_pieces, anchor // num_pieces])
    offset_start = anchor_pos[:2] - anchor_pos_in_puzzle
    drawing_correctness = np.zeros((num_pieces, num_pieces), dtype=np.uint8)
    num_correct_pieces = 0
    for j in range(num_pieces*num_pieces):
        estimated_pos_piece = np.unravel_index(np.argmax(p_final[:,:,0,j]), p_final[:,:,0,j].shape)
        pos_y = j % num_pieces
        pos_x = j // num_pieces
        correct_position = offset_start + np.asarray([pos_x, pos_y])
        print(f"piece {j} = estimated: {estimated_pos_piece}, correct: {correct_position}")
        #pdb.set_trace()
        if np.isclose(np.sum(np.abs(np.subtract(estimated_pos_piece, correct_position))), 0):
            num_correct_pieces += 1
            drawing_correctness[pos_x, pos_y] = (255)

    #print(num_correct_pieces)
    
    plt.figure(figsize=(32,32))
    Y, X, Z, _ = p_final.shape
    pieces_folder = os.path.join(f"{fnames.output_dir}_{cfg.num_patches_side}x{cfg.num_patches_side}", args.dataset, args.puzzle, f"{fnames.pieces_folder}")
    pieces_files = os.listdir(pieces_folder)
    pieces_files.sort()
    pieces_excl = []
    all_pieces = np.arange(len(pieces_files))
    pieces = [p for p in all_pieces if p not in all_pieces[pieces_excl]]
    sol = get_sol_from_p(p_final=p_final)
    plt.subplot(131)
    plt.title('reference image', fontsize=32)
    im_ref = plt.imread(os.path.join(fnames.data_path, args.dataset, fnames.images_folder, f"{args.puzzle}.jpg"))
    plt.imshow(im_ref, cmap='gray')
    fin_im1 = reconstruct_puzzle(sol, Y, X, pieces, pieces_files, pieces_folder)
    start_point = offset_start*cfg.piece_size
    end_point = start_point + num_pieces*cfg.piece_size
    solution_img = fin_im1[start_point[0]:end_point[0], start_point[1]:end_point[1]]
    plt.subplot(133)
    plt.title(f'solution of the puzzle ({num_correct_pieces / (num_pieces**2) * 100:.03f} %)', fontsize=32)
    plt.imshow(solution_img)
    plt.subplot(132)
    plt.title(f'correct pieces (yellow): {num_correct_pieces}', fontsize=32)
    plt.imshow(drawing_correctness)
    if args.visualize is True:
        plt.show()
    else:
        outputpath = os.path.join(args.output, 'visualization_solution.png')
        plt.tight_layout()
        plt.savefig(outputpath)
        #pdb.set_trace()
    return 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='........ ')  # add some discription
    parser.add_argument('--dataset', type=str, default='manual_lines', help='dataset folder')   # repair, wikiart, manual_lines, architecture
    parser.add_argument('-pz', '--puzzle', type=str, default='lines1', help='puzzle folder')           # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-o', '--output', type=str, default='', help='puzzle folder')           # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-s', '--solution', type=str, default='solution.mat', help='path to the final solution matrix')           # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-pc', '--pieces', type=int, default=8, help='number of pieces (per side)')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    parser.add_argument('-v', '--visualize', default=False, action='store_true', help='use to show the solution')                 # repair_g28, aki-kuroda_night-2011, pablo_picasso_still_life
    args = parser.parse_args()

    main(args)