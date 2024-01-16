""" 
It draws the solution from the p_final matrix.
It creates two images, one from the pieces and one using the "only_lines" images, to see what the solver "saw"
"""
import argparse, os, json
import matplotlib.pyplot as plt
import numpy as np 
from configs import folder_names as fnames
from puzzle_utils.visualization import reconstruct_puzzle
from puzzle_utils.pieces_utils import calc_parameters
from puzzle_utils.shape_utils import prepare_pieces_v2
import pdb 
def main(args):

    if args.puzzle == '':  
        puzzles = os.listdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset))
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset,puz)) is True]
    else:
        puzzles = [args.puzzle]

    print(f"\nWill draw the solution for for: {puzzles}\n")
    for puzzle in puzzles:

        puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle)
        solutions_folders = [sol_fld for sol_fld in os.listdir(puzzle_root_folder) if fnames.solution_folder_name in sol_fld]
        print(f"found {len(solutions_folders)} solutions:", solutions_folders)
        for solution_folder in solutions_folders:
            
            print("working on:", solution_folder)
            cmp_parameter_path = os.path.join(puzzle_root_folder, 'compatibility_parameters.json')
            if os.path.exists(cmp_parameter_path):
                print("never tested! remove this comment afterwars (line 21)")
                with open(cmp_parameter_path, 'r') as cp:
                    ppars = json.load(cmp_parameter_path)
            else:
                pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, puzzle, verbose=True)
                ppars = calc_parameters(img_parameters)
            
            p_final_path = os.path.join(puzzle_root_folder, solution_folder, 'p_final.npy')
            p_final_voodoo_0darray = np.load(p_final_path, allow_pickle=True)
            p_final = p_final_voodoo_0darray.item()['p_final']
            pieces_folder = os.path.join(puzzle_root_folder, fnames.pieces_folder)
            solution_drawing_pieces = reconstruct_puzzle(p_final, pieces_folder, ppars)
            solution_drawing_pieces = np.clip(solution_drawing_pieces, 0, 1)
            rec_sol_path = os.path.join(puzzle_root_folder, solution_folder, 'sol_rec_from_p_final.jpg')
            plt.imsave(rec_sol_path, solution_drawing_pieces)
            print('done!\n')
            # only lines
            # only_lines_folder = os.path.join(puzzle_root_folder, fnames.lines_output_name, args.method, 'lines_only')
            # solution_drawing_only_lines = reconstruct_puzzle(p_final, only_lines_folder, ppars, suffix='_t')
            # solution_drawing_only_lines = np.clip(solution_drawing_only_lines, 0, 1)
            # rec_sol_l_path = os.path.join(puzzle_root_folder, solution_folder, 'sol_rec_from_p_final_lines_only.jpg')
            # plt.imsave(rec_sol_l_path, solution_drawing_pieces)
            # print('done with', solution_folder)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Draw the solution from the matrix')  # add some description
    parser.add_argument('--dataset', type=str, default='synthetic_irregular_4_pieces_by_drawing_lines_ccayvh', help='dataset folder')
    parser.add_argument('--puzzle', type=str, default='', help='puzzle folder (leave empty to go through the whole dataset)')
    parser.add_argument('--method', type=str, default='exact', help='method used for compatibility')  # exact, deeplsd

    args = parser.parse_args()

    main(args)