import numpy as np
from scipy.io import savemat 
import argparse 
import pdb
import matplotlib.pyplot as plt 
import cv2
import json, os 
from PIL import Image 

#from configs import repair_cfg as cfg
from configs import folder_names as fnames

from puzzle_utils.shape_utils import prepare_pieces_v2, create_grid, shape_pairwise_compatibility, \
    get_outside_borders, place_on_canvas, get_borders_around, include_shape_info
from puzzle_utils.pieces_utils import calc_parameters_v2
from puzzle_utils.visualization import save_vis

def main(args):

    if args.puzzle == '':  
        puzzles = os.listdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset))
        puzzles = [puz for puz in puzzles if os.path.isdir(os.path.join(os.getcwd(), fnames.output_dir, args.dataset,puz)) is True]
    else:
        puzzles = [args.puzzle]

    print(f"\nWill calculate regions masks for: {puzzles}\n")
    for puzzle in puzzles:

        ######
        # PREPARE PIECES AND GRIDS
        # 
        # pieces is a list of dictionaries with the pieces (and mask, cm, id)
        # img_parameters contains the size of the image and of the pieces
        # ppars contains all the values needed for computing stuff (p_hs, comp_range..)
        # ppars is a dict but can be accessed by pieces_paramters.property!
        print()
        print("-" * 50)
        print(puzzle)
        pieces, img_parameters = prepare_pieces_v2(fnames, args.dataset, puzzle, args.num_pieces, verbose=True)
        # PARAMETERS
        puzzle_root_folder = os.path.join(os.getcwd(), fnames.output_dir, args.dataset, puzzle)
        cmp_parameter_path = os.path.join(puzzle_root_folder, 'compatibility_parameters.json')
        # if os.path.exists(cmp_parameter_path):
        #     print("never tested! remove this comment afterwars (line 53 of comp_irregular.py)")
        #     with open(cmp_parameter_path, 'r') as cp:
        #         ppars = json.load(cp)
        # else:
        ppars = calc_parameters_v2(img_parameters, args.xy_step, args.xy_grid_points, args.theta_step)
        # for ppk in ppars.keys():
        #     if type(ppars[ppk])== np.uint8:
        #         ppars[ppk] = int(ppars[ppk])
        #     print(ppk, ":", type(ppars[ppk]))
        # pdb.set_trace()
        with open(cmp_parameter_path, 'w') as cpj:
            json.dump(ppars, cpj, indent=3)
        print("saved json compatibility file")

        # INCLUDE SHAPE
        pieces = include_shape_info(fnames, pieces, args.dataset, puzzle, args.method, line_thickness=3)

        grid_size_xy = ppars.comp_matrix_shape[0]
        grid_size_rot = ppars.comp_matrix_shape[2]
        #grid, grid_step_size = create_grid(grid_size_xy, ppars.p_hs, ppars.canvas_size)

        print()
        print('#' * 50)
        print('SETTINGS')
        print(f"The puzzle (maybe rescaled) has size {ppars.img_size[0]}x{ppars.img_size[1]} pixels")
        print(f'Pieces are squared images of {ppars.piece_size}x{ppars.piece_size} pixels (p_hs={ppars.p_hs})')
        print(f"This puzzle has {ppars.num_pieces} pieces")
        print(f'The region matrix has shape: [{grid_size_xy}, {grid_size_xy}, {grid_size_rot}, {len(pieces)}, {len(pieces)}]')
        print(f'Using a grid on xy and {grid_size_rot} rotations on {len(pieces)} pieces')
        print(f'\txy_step: {ppars.xy_step}, rot_step: {ppars.theta_step}')
        print(f'Canvas size: {ppars.canvas_size}x{ppars.canvas_size}')
        print('#' * 50)
        print()

        ## CREATE MATRIX                      
        RM_combo = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        RM_lines = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        RM_shapes = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
        for i in range(len(pieces)):
            for j in range(len(pieces)):
                print(f"regions for pieces {i:>2} and {j:>2}", end='\r')
                if i == j:
                    RM_combo[:,:,0,j,i] = -1
                    RM_lines[:,:,0,j,i] = -1
                    RM_shapes[:,:,0,j,i] = -1
                else:
                    RM_combo[:,:,0,j,i] = np.asarray([[0, 1, 0],[1, -1, 1], [0, 1, 0]])
                    RM_lines[:,:,0,j,i] = np.asarray([[0, 1, 0],[1, -1, 1], [0, 1, 0]])
                    RM_shapes[:,:,0,j,i] = np.asarray([[0, 1, 0],[1, -1, 1], [0, 1, 0]])
                  
        print("\n")
        print('Done calculating')
        print('#' * 50)
        print('Saving the matrix..')     
        if args.num_pieces == 8:
            output_root_dir = f"{fnames.output_dir}_8x8"
        else:
            output_root_dir = fnames.output_dir
        output_folder = os.path.join(output_root_dir, args.dataset, puzzle, fnames.rm_output_name)
        # should we add this to the folder? it will create a subfolder that we may not need
        # f"{ppars.rm_output_dir}_{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}")
        vis_folder = os.path.join(output_folder, fnames.visualization_folder_name)
        os.makedirs(vis_folder, exist_ok=True)
        RM_D = {}
        RM_D['RM'] = RM_combo
        RM_D['RM_lines'] = RM_lines
        RM_D['RM_shapes'] = RM_shapes

        filename = f'{output_folder}/RM_{puzzle}'
        savemat(f'{filename}.mat', RM_D)
        if args.save_visualization is True:
            print('Creating visualization')
            save_vis(RM_combo, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_combo_{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'), f"regions matrix {puzzle}", save_every=4, all_rotation=False)
            save_vis(RM_lines, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_lines_{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'), f"overlap {puzzle}", save_every=4, all_rotation=False)
            save_vis(RM_shapes, pieces, ppars.theta_step, os.path.join(vis_folder, f'visualization_shapes_{puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'), f"borders {puzzle}", save_every=4, all_rotation=False)
        print(f'Done with {puzzle}\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computing compatibility matrix')
    parser.add_argument('--dataset', type=str, default='', help='dataset (name of the folders)')
    parser.add_argument('--puzzle', type=str, default='', help='puzzle to work on - leave empty to generate for the whole dataset')
    parser.add_argument('--method', type=str, default='exact', help='method line detection')  # exact, manual, deeplsd
    parser.add_argument('--save_everything', type=bool, default=False, help='save also overlap and borders matrices')
    parser.add_argument('--save_visualization', type=bool, default=True, help='save an image that showes the matrices color-coded')
    parser.add_argument('-np', '--num_pieces', type=int, default=0, help='number of pieces (per side) - use 0 (default value) for synthetic pieces')  # 8
    parser.add_argument('--xy_step', type=int, default=32, help='the step (in pixels) between each grid point')
    parser.add_argument('--xy_grid_points', type=int, default=3, 
        help='the number of points in the grid (for each axis, total number will be the square of what is given)')
    parser.add_argument('--theta_step', type=int, default=0, help='degrees of each rotation')
    parser.add_argument('--DEBUG', action='store_true', default=False, help='WARNING: will use debugger! It stops and show the matrices!')
    args = parser.parse_args()
    main(args)

