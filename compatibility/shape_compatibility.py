from puzzle_utils.shape_utils import prepare_pieces, create_grid, shape_pairwise_compatibility
from puzzle_utils.regions import read_region_masks
import numpy as np
import scipy
import argparse 
import pdb
import matplotlib.pyplot as plt 
import cv2
import json, os 
from configs import repair_cfg as cfg
from configs import folder_names as fnames
from rpf_utils.visualization import save_vis
from numba import jit

@jit (parallel=True)
def compute_compatibility_in_parallel(pieces, CM, RM, cfg, grid, urm: bool):

    grid_size_rot = CM.shape[2]
    for i in range(len(pieces)):
        for j in range(len(pieces)):
            if i == j:
                CM[:,:,:,i,j] = -1
            else:
                if urm:
                    #print(i, j)
                    for t in range(grid_size_rot):
                        print(i, j, t)
                        idx = np.where(RM[:, :, t, i, j] > 0)
                        for x, y in zip(idx[0], idx[1]):
                            shape_comp = shape_pairwise_compatibility(pieces[i], pieces[j], int(y), int(x), t, cfg, grid, sigma=cfg.sigma)
                            CM[x, y, t, j, i] = shape_comp
                            #print(f"C({x:>2}, {y:>2}, {t:>2}, {j:>2}, {i:>2}) = {shape_comp:>8.3f}", end='\r')
                else:
                    for x in range(grid_size_xy):
                        for y in range(grid_size_xy):
                            for t in range(grid_size_rot):
                                # HERE WE COMPUTE THE COMPATIBILITIES
                                shape_comp = shape_pairwise_compatibility(pieces[i], pieces[j], y, x, t, cfg, grid, sigma=cfg.sigma)
                                CM[x, y, t, j, i] = shape_comp
    return CM 

def main(args):

    ## PREPARE PIECES AND GRIDS
    pieces = prepare_pieces(cfg, fnames, args.puzzle)
    grid_size_xy = cfg.comp_matrix_shape[0]
    grid_size_rot = cfg.comp_matrix_shape[2]
    # grid, grid_step_size = create_grid(grid_size_xy, cfg.p_hs, cfg.canvas_size)
    if args.urm is True:
        ok, RM = read_region_masks(cfg, args.puzzle)
        if ok < 0:
            print("Error (read above), stopping here")
            return 

    output_folder = os.path.join(cfg.output_dir, args.puzzle, cfg.cm_output_name)
    os.makedirs(output_folder, exist_ok=True)
    # f"{cfg.cm_output_dir}_{args.puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}")
    vis_folder = os.path.join(output_folder, cfg.visualization_folder_name)
    os.makedirs(vis_folder, exist_ok=True)

    print('#' * 50)
    print('SETTINGS')
    print(f'CM has shape: [{grid_size_xy}, {grid_size_xy}, {grid_size_rot}, {len(pieces)}, {len(pieces)}]')
    print(f'Using a grid  on xy and {grid_size_rot} rotations on {len(pieces)} pieces')
    print(f'Pieces are squared images of {cfg.piece_size}x{cfg.piece_size} pixels (p_hs={cfg.p_hs})')
    print(f'xy_step: {cfg.xy_step}, rot_step: {cfg.theta_step}')
    print(f'Canvas size: {cfg.canvas_size}x{cfg.canvas_size}')
    print(f'Using regions matrix: {args.urm}')
    print(f'Output folder: {output_folder}')
    print('#' * 50)
    
    ## CREATE MATRIX
    CM = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
    
    if args.parallel:
        CM = compute_compatibility_in_parallel(pieces, CM, RM, cfg, grid, args.urm)
    else:
        ## COMPUTE SCORES
        print('Calculations.. (this may take a while)')
        for i in range(len(pieces)):
            for j in range(len(pieces)):
                if i == j:
                    CM[:,:,:,i,j] = -1
                else:
                    if args.urm:
                        for t in range(grid_size_rot):
                            idx = np.where(RM[:, :, t, i, j] > 0)
                            for x, y in zip(idx[0], idx[1]):
                                shape_comp = shape_pairwise_compatibility(pieces[i], pieces[j], int(y), int(x), t, cfg, grid, sigma=cfg.sigma)
                                CM[x, y, t, j, i] = shape_comp
                                print(f"C({x:>2}, {y:>2}, {t:>2}, {j:>2}, {i:>2}) = {shape_comp:>8.3f}", end='\r')
                    else:
                        for x in range(grid_size_xy):
                            for y in range(grid_size_xy):
                                for t in range(grid_size_rot):
                                    # HERE WE COMPUTE THE COMPATIBILITIES
                                    shape_comp = shape_pairwise_compatibility(pieces[i], pieces[j], y, x, t, cfg, grid, sigma=cfg.sigma)
                                    CM[x, y, t, j, i] = shape_comp
                                    print(f"C({x:>2}, {y:>2}, {t:>2}, {j:>2}, {i:>2}) = {shape_comp:>8.3f}", end='\r')
    
    print()
    print('Done calculating')
    print('#' * 50)
    print('Saving the matrix..')
    #pdb.set_trace()
    ## SAVE AS .mat FILES
    CM_D = {}
    CM_D['R'] = CM
    
    filename = os.path.join(output_folder, 'CM_shape.mat')
    scipy.io.savemat(filename, CM_D)

    if cfg.save_visualization is True:
        print('Creating visualization')
        save_vis(CM, pieces, os.path.join(vis_folder, f'visualization_{args.puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'), f"comp matrix {args.puzzle}", all_rotation=True)

    print("Finished!")
    print(f"Saved in {output_folder}")
    print('#' * 50)
    #pdb.set_trace()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computing compatibility matrix')
    parser.add_argument('--puzzle', type=str, default='repair_g28', help='puzzle to work on')
    parser.add_argument('--urm', '-r',  action='store_true')
    parser.add_argument('--parallel', '-p', action='store_true')
    args = parser.parse_args()
    main(args)