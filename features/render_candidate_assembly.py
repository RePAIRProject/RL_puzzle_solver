"""
1. read two pieces
2. make RM 
3. iterate over valid points
    - render each position 
"""
import json
import os 
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from puzzle_utils import shape_utils as shu
from puzzle_utils import rm_utils as rmu
import argparse 

class CfgParameters(dict):
    __getattr__ = dict.__getitem__

def load_as_cfg(path):
    with open(args.pars, 'r') as jpf:
        pars_dict = json.load(jpf)
    ppars = CfgParameters()
    for pdk in pars_dict.keys():
        ppars[pdk] = pars_dict[pdk]
    return ppars

def main(args):

    print('Loading pieces')
    piece_i = shu.prepare_piece_v2(args.i)
    piece_j = shu.prepare_piece_v2(args.j)
    ppars = load_as_cfg(args.pars)
    print("calculating RM")
    RM_ij = rmu.compute_pairwise_shape_based_RM(piece_i, piece_j, ppars, dilate=False, erode=True)
    if ppars.motif_based == True:
        print("NOT DONE YET!\nTODO:")
        print("RM_ij_motifs = rmu.compute_pairwise_motif_based_RM(piece_i, piece_j, ppars, dilate=False, erode=True)")
        breakpoint()
    if args.o == "":
        output_folder = f"candidate_assembly_{args.i.split('/')[-1][:-4]}_{args.j.split('/')[-1][:-4]}"
    else: 
        output_folder = args.o 
    os.makedirs(output_folder, exist_ok=True)
    print("Saving visualization RM")
    plt.figure(figsize=(32, 18))#; plt.title("Candidate Assembly Overview")
    plt.subplot(241); plt.title('Image of piece i')
    plt.imshow(piece_i['img'])
    plt.subplot(242); plt.title('Mask of piece i')
    plt.imshow(piece_i['mask'])
    plt.subplot(243); plt.title('Image of piece j')
    plt.imshow(piece_j['img'])
    plt.subplot(244); plt.title('Mask of piece j')
    plt.imshow(piece_j['mask'])
    for s in range(ppars.theta_grid_points): 
        plt.subplot(2,4,5+s); plt.title(f'RM (rot={s*ppars.theta_step})')
        plt.imshow(RM_ij[:,:,s])
    plt.savefig(os.path.join(output_folder, 'visualization_RM.png'))
    grid, xy_step = shu.create_grid_v2(ppars)
    assembly_images_folder = os.path.join(output_folder, 'assembly_images')
    os.makedirs(assembly_images_folder, exist_ok=True)
    print("Rendering single assembly transformation")
    for k in range(ppars.xy_grid_points):
        for l in range(ppars.xy_grid_points):
            for t in range(ppars.theta_grid_points):
                if RM_ij[k,l,t] > 0:
                    xy_coords = grid[k,l]
                    rot_coords = t * ppars.theta_step
                    image = shu.render_pair_at(piece_i, piece_j, ppars, (xy_coords[0], xy_coords[1], rot_coords))
                    cv2.imwrite(os.path.join(assembly_images_folder, f"assembly_{int(xy_coords[0]):02d}_{int(xy_coords[1]):02d}_{rot_coords:02d}.png"), image)
                    print(f"rendering image assembly_{xy_coords[0]}_{xy_coords[1]}_{rot_coords}.png", end='\r')
    
    print('\nDone!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render to image candidate assembly transformation')
    parser.add_argument('-pars', type=str, default="", 
                        help='path to the json parameter file!')
    parser.add_argument('-i', type=str, default="RPf_00194", 
                        help='first player (in the center)')             
    parser.add_argument('-j', type=str, default="RPf_00197", 
                        help='second player (moving around)')  
    parser.add_argument('-o', type=str, default="", 
                        help='output folder')              
    args = parser.parse_args()
    main(args)