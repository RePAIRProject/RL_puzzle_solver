



from rpf_utils.shape_utils import prepare_pieces, create_grid, shape_pairwise_compatibility
import numpy as np
import scipy
import argparse 
import pdb
import matplotlib.pyplot as plt 
import cv2
import json, os 
from configs import rp_cfg as cfg
from rpf_utils.visualization import save_vis

def main(args):

    ## PREPARE PIECES AND GRIDS
    #pdb.set_trace()
    pieces = prepare_pieces(cfg, args.puzzle)
    grid_size_xy = cfg.comp_matrix_shape[0]
    grid_size_rot = cfg.comp_matrix_shape[2]
    grid, grid_step_size = create_grid(grid_size_xy, cfg.p_hs, cfg.canvas_size)
    print('#' * 50)
    print('SETTINGS')
    print(f'CM has shape: [{grid_size_xy}, {grid_size_xy}, {grid_size_rot}, {len(pieces)}, {len(pieces)}]')
    print(f'Using a grid  on xy and {grid_size_rot} rotations on {len(pieces)} pieces')
    print(f'Pieces are squared images of {cfg.piece_size}x{cfg.piece_size} pixels (p_hs={cfg.p_hs})')
    print(f'xy_step: {cfg.xy_step}, rot_step: {cfg.theta_step}')
    print(f'Canvas size: {cfg.canvas_size}x{cfg.canvas_size}')
    print('#' * 50)
    #pdb.set_trace()
    ## CREATE MATRIX
    CM = np.zeros((grid_size_xy, grid_size_xy, grid_size_rot, len(pieces), len(pieces)))
    ## COMPUTE SCORES
    print('Calculations.. (this may take a while)')
    for i in range(len(pieces)):
        for j in range(len(pieces)):
            if i == j:
                CM[:,:,:,i,j] = -1
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
    
    output_folder = os.path.join(cfg.cm_output_dir, f"{cfg.cm_output_dir}_{args.puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}")
    vis_folder = os.path.join(output_folder, cfg.visualization_folder_name)
    os.makedirs(vis_folder, exist_ok=True)

    filename = f'{output_folder}/CM_shape_{args.puzzle}_{grid_size_xy}x{grid_size_xy}x{grid_size_rot}x{len(pieces)}x{len(pieces)}'
    scipy.io.savemat(f'{filename}.mat', CM_D)

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
    args = parser.parse_args()
    main(args)